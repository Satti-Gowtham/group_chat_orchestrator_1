from dotenv import load_dotenv
from group_chat_orchestrator.schemas import InputSchema
import uuid
from typing import Dict, Any
from naptha_sdk.schemas import KBRunInput, OrchestratorRunInput, OrchestratorDeployment, AgentRunInput
from naptha_sdk.modules.agent import Agent
from naptha_sdk.modules.kb import KnowledgeBase
from naptha_sdk.user import sign_consumer_id, get_private_key_from_pem
from naptha_sdk.utils import get_logger
from naptha_sdk.inference import InferenceClient
import json
import os
import traceback
from group_chat_orchestrator.text_processing import (
    clean_text,
    process_agent_response,
    get_relevant_context,
    summarize_questions_to_topic
)
from group_chat_orchestrator.utils import save_agent_results
from group_chat_orchestrator.prompts import (
    create_research_prompt,
    create_agent_prompt,
)

load_dotenv()
logger = get_logger(__name__)

class GroupChatOrchestrator:
    async def create(self, deployment: OrchestratorDeployment, *args, **kwargs):
        """Initialize the orchestrator with all agents and knowledge base."""
        self.deployment = deployment
        self.agent_deployments = self.deployment.agent_deployments
        
        # Initialize all agents
        self.researcher_agent = Agent()  # Initial research and topic exploration
        self.analyst_agent = Agent()     # Analysis and deeper insights
        self.synthesizer_agent = Agent() # Synthesis and final conclusions
        self.inference_client = InferenceClient(deployment.node)
        
        # Create all agents
        await self.researcher_agent.create(deployment=self.agent_deployments[0])
        await self.analyst_agent.create(deployment=self.agent_deployments[1])
        await self.synthesizer_agent.create(deployment=self.agent_deployments[2])
        
        # Initialize knowledge base
        self.groupchat_kb = KnowledgeBase()
        await self.groupchat_kb.create(deployment=self.deployment.kb_deployments[0])

    async def run(self, module_run: OrchestratorRunInput) -> Dict[str, Any]:
        """Run the group chat orchestration with multiple agents."""
        try:
            # Generate a unique run ID
            run_id = str(uuid.uuid4())
            kb_deployment = self.deployment.kb_deployments[0]
            current_topic = clean_text(module_run.inputs.topic)

            # First round: Research Agent
            research_prompt = create_research_prompt(current_topic)
            research_result = await self.researcher_agent.run(AgentRunInput(
                consumer_id=module_run.consumer_id,
                inputs={
                    "topic": current_topic,
                    "round": 1,
                    "context": {
                        "previous_questions": [],  # Initial research has no previous questions
                        "relevant_findings": [],    # Initial research has no previous findings
                        "formatted_content": clean_text(research_prompt)
                    },
                    "temperature": module_run.inputs.temperature,
                    "max_tokens": module_run.inputs.max_tokens
                },
                deployment=self.agent_deployments[0],
                signature=module_run.signature
            ))

            try:
                research_data = process_agent_response(research_result.results[-1])
                logger.info(f"Processed research data: {json.dumps(research_data, indent=2)}")
                # Save research results
                save_agent_results(
                    run_id, 
                    "researcher", 
                    research_result.results[-1], 
                    research_data
                )
            except Exception as e:
                logger.error(f"Error processing research results: {str(e)}")
                research_data = {
                    "findings": [],
                    "questions": [],
                    "metadata": {"error": str(e)}
                }

            # Store research results in KB
            kb_input = {
                "run_id": run_id,
                "topic": current_topic,
                "findings": research_data["findings"],
                "questions": research_data["questions"],
                "metadata": {
                    "round": 1,
                    "topic": current_topic,
                    "type": "research"
                }
            }
            logger.info(f"Storing research in KB: {json.dumps(kb_input, indent=2)}")
            
            await self.groupchat_kb.run(KBRunInput(
                consumer_id=module_run.consumer_id,
                inputs={
                    "func_name": "add_data",
                    "func_input_data": kb_input
                },
                deployment=kb_deployment,
                signature=module_run.signature
            ))

            # Process subsequent agents
            all_findings = research_data.get("findings", [])
            all_questions = research_data.get("questions", [])
            
            # Summarize questions into a focused topic for subsequent agents
            summarised_topic = await summarize_questions_to_topic(
                all_questions, 
                current_topic,
                self.inference_client,
                self.agent_deployments[0].config
            )

            agent_findings = research_data["findings"]
            logger.info(f"Summarized topic for subsequent agents: {summarised_topic}")
            
            for round_num, (agent, agent_name) in enumerate([
                (self.analyst_agent, "analyst"),
                (self.synthesizer_agent, "synthesizer")
            ], start=2):
                # Get context from KB based on summarized topic
                context = await get_relevant_context(
                    self.groupchat_kb,
                    kb_deployment,
                    run_id,
                    summarised_topic,
                    module_run.consumer_id,
                    module_run.signature
                )

                # Format the agent's prompt to directly address previous questions
                agent_prompt = create_agent_prompt(
                    current_topic=current_topic,
                    summarised_topic=summarised_topic,
                    agent_name=agent_name,
                    context=context
                )
                
                # Pass the prompt and context in the same format as researcher agent
                agent_result = await agent.run(AgentRunInput(
                    consumer_id=module_run.consumer_id,
                    inputs={
                        "topic": current_topic,
                        "round": round_num,
                        "context": {
                            "previous_questions": context["previous_questions"],
                            "relevant_findings": context["relevant_findings"],
                            "formatted_content": clean_text(agent_prompt)
                        },
                        "temperature": module_run.inputs.temperature,
                        "max_tokens": module_run.inputs.max_tokens
                    },
                    deployment=self.agent_deployments[round_num-1],
                    signature=module_run.signature
                ))

                try:
                    agent_data = process_agent_response(agent_result.results[-1])
                    logger.info(f"Processed {agent_name} data: {json.dumps(agent_data, indent=2)}")
                    # Save agent results with prompt
                    save_agent_results(
                        run_id, 
                        agent_name, 
                        agent_result.results[-1], 
                        agent_data,
                        prompt=agent_prompt
                    )

                    # Update summarized topic for next agent based on new questions
                    summarised_topic = await summarize_questions_to_topic(
                        agent_data["questions"],
                        current_topic,
                        self.inference_client,
                        self.agent_deployments[0].config
                    )
                    logger.info(f"Updated summarized topic for next agent: {summarised_topic}")
                except Exception as e:
                    logger.error(f"Error processing {agent_name} results: {str(e)}")
                    agent_data = {
                        "findings": [],
                        "questions": [],
                        "metadata": {"error": str(e)}
                    }

                # Store agent results in KB
                kb_input = {
                    "run_id": run_id,
                    "topic": current_topic,
                    "findings": agent_data["findings"],
                    "questions": agent_data["questions"],
                    "metadata": {
                        "round": round_num,
                        "topic": current_topic,
                        "type": agent_name
                    }
                }
                logger.info(f"Storing {agent_name} in KB: {json.dumps(kb_input, indent=2)}")
                
                await self.groupchat_kb.run(KBRunInput(
                    consumer_id=module_run.consumer_id,
                    inputs={
                        "func_name": "add_data",
                        "func_input_data": kb_input
                    },
                    deployment=kb_deployment,
                    signature=module_run.signature
                ))

                # Update combined findings and questions
                if agent_data.get("findings"):
                    all_findings.extend(agent_data["findings"])
                if agent_data.get("questions"):
                    all_questions.extend(agent_data["questions"])

            return {
                "status": "success",
                "findings": all_findings,
                "questions": all_questions,
                "metadata": {
                    "run_id": run_id,
                    "num_rounds": 3,
                    "final_topic": current_topic
                }
            }

        except Exception as e:
            logger.error(f"Error in group chat orchestration: {str(e)}")
            logger.error(f"Full traceback:\n{''.join(traceback.format_tb(e.__traceback__))}")
            return {
                "status": "error",
                "message": str(e)
            }

async def run(module_run: Dict):
    """Main entry point for the module."""
    module_run = OrchestratorRunInput(**module_run)
    module_run.inputs = InputSchema(**module_run.inputs)
    group_chat_orchestrator = GroupChatOrchestrator()
    await group_chat_orchestrator.create(module_run.deployment)
    return await group_chat_orchestrator.run(module_run)

if __name__ == "__main__":
    import asyncio
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import setup_module_deployment

    naptha = Naptha()

    deployment = asyncio.run(setup_module_deployment("group_chat_orchestrator", "group_chat_orchestrator/configs/deployment.json", node_url = os.getenv("NODE_URL")))

    inputs_dict = {
        "run": {
            "func_name": "run",
            "func_input_data": {
                "topic": "What are the implications of synthetic life?",
                "num_rounds": 3,
                "temperature": 0.7,
                "max_tokens": 2000
            },
        }
    }

    module_run = {
        "inputs": inputs_dict["run"],
        "deployment": deployment,
        "consumer_id": naptha.user.id,
        "signature": sign_consumer_id(naptha.user.id, get_private_key_from_pem(os.getenv("PRIVATE_KEY")))
    }

    response = asyncio.run(run(module_run))
    print("Response: ", response) 