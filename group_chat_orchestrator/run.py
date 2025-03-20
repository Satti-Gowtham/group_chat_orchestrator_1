import asyncio
import logging
import json
import os
from pathlib import Path
from typing import List, Dict, Any
from naptha_sdk.modules.agent import Agent
from naptha_sdk.modules.kb import KnowledgeBase
from naptha_sdk.schemas import OrchestratorDeployment, OrchestratorRunInput, KBRunInput, AgentRunInput, NodeConfig
from naptha_sdk.user import sign_consumer_id, get_private_key_from_pem
from naptha_sdk.client.naptha import Naptha
from naptha_sdk.configs import setup_module_deployment

from schemas import ResearchInput, ResearchOutput
from utils import get_logger, ensure_output_dir, write_output

logger = get_logger(__name__)

class GroupChatOrchestrator:
    async def create(self, deployment: OrchestratorDeployment, *args, **kwargs):
        """Create the orchestrator with the given deployment configuration."""
        self.deployment = deployment
        self.agent_deployments = self.deployment.agent_deployments
        
        # Initialize all agents
        self.agents = []
        for i in range(len(self.agent_deployments)):
            agent = Agent()
            await agent.create(deployment=self.agent_deployments[i], *args, **kwargs)
            self.agents.append(agent)
        
        # Initialize knowledge base
        # self.research_kb = KnowledgeBase()
        # await self.research_kb.create(deployment=self.deployment.kb_deployments[0], *args, **kwargs)

    async def run(self, module_run: OrchestratorRunInput, *args, **kwargs):
        """Run the research orchestration between agents."""
        run_id = str(uuid.uuid4())
        # kb_deployment = self.deployment.kb_deployments[0]
        
        # Create KB table
        # await self.research_kb.run(KBRunInput(
        #     consumer_id=module_run.consumer_id,
        #     inputs={"func_name": "initialize"},
        #     deployment=kb_deployment,
        #     signature=sign_consumer_id(module_run.consumer_id, get_private_key_from_pem(os.getenv("PRIVATE_KEY")))
        # ))

        # Initialize research topic
        research_topic = module_run.inputs.research_topic
        
        # Create output directory
        output_dir = Path("research_outputs")
        ensure_output_dir(str(output_dir))

        all_findings = []
        
        # Run research with each agent
        for agent_num, agent in enumerate(self.agents):
            try:
                # Run agent research
                agent_run_input = AgentRunInput(
                    consumer_id=module_run.consumer_id,
                    inputs={
                        "topic": research_topic
                    },
                    deployment=self.agent_deployments[agent_num],
                    signature=sign_consumer_id(module_run.consumer_id, get_private_key_from_pem(os.getenv("PRIVATE_KEY")))
                )
                
                response = await agent.run(agent_run_input)
                
                # Process agent's findings
                if response.results:
                    finding = response.results[-1]
                    all_findings.append(finding)
                    
                    # Store in knowledge base
                    # await self.research_kb.run(KBRunInput(
                    #     consumer_id=module_run.consumer_id,
                    #     inputs={
                    #         "func_name": "ingest_knowledge",
                    #         "func_input_data": {
                    #             "run_id": run_id,
                    #             "topic": research_topic,
                    #             "content": finding,
                    #             "agent_id": f"agent_{agent_num}"
                    #         }
                    #     },
                    #     deployment=kb_deployment,
                    #     signature=sign_consumer_id(module_run.consumer_id, get_private_key_from_pem(os.getenv("PRIVATE_KEY")))
                    # ))
                    
                    # Write individual findings
                    write_output(
                        finding,
                        str(output_dir / f"finding_agent{agent_num}.txt")
                    )
            
            except Exception as e:
                logger.error(f"Error with agent {agent_num}: {e}")
                raise e

        # Generate final summary
        summary_prompt = f"Summarize the following research findings about {research_topic}:\n\n"
        summary_prompt += "\n\n".join(all_findings)

        summary_agent = self.agents[0]
        summary_response = await summary_agent.run(AgentRunInput(
            consumer_id=module_run.consumer_id,
            inputs={
                "tool_name": "summarize",
                "tool_input_data": summary_prompt
            },
            deployment=self.agent_deployments[0],
            signature=sign_consumer_id(module_run.consumer_id, get_private_key_from_pem(os.getenv("PRIVATE_KEY")))
        ))

        summary = summary_response.results[-1] if summary_response.results else "No summary generated"
        
        # Write summary
        write_output(summary, str(output_dir / "summary.txt"))

        return ResearchOutput(
            structured_report=str(output_dir / "finding_agent0.txt"),
            summary=str(output_dir / "summary.txt"),
            findings=str(output_dir / "finding_agent0.txt")
        )

async def run(module_run: Dict, *args, **kwargs):
    """Main entry point for the module."""
    try:
        module_run = OrchestratorRunInput(**module_run)
        module_run.inputs = ResearchInput(**module_run.inputs)
        orchestrator = GroupChatOrchestrator()
        await orchestrator.create(module_run.deployment)
        result = await orchestrator.run(module_run)
        return result
    except Exception as e:
        logger.error(f"Error in group_chat_orchestrator run: {e}")
        raise e

if __name__ == "__main__":
    import uuid
    from dotenv import load_dotenv
    load_dotenv()

    naptha = Naptha()

    async def test_orchestrator():
        deployment = await setup_module_deployment(
            "orchestrator",
            "group_chat_orchestrator/configs/deployment.json",
            node_url=os.getenv("NODE_URL")
        )

        test_run_input = {
            "deployment": deployment,
            "consumer_id": naptha.user.id,
            "signature": sign_consumer_id(naptha.user.id, get_private_key_from_pem(os.getenv("PRIVATE_KEY"))),
            "inputs": {
                "research_topic": "AI in healthcare",
                "num_agents": 3,
                "use_roles": False
            }
        }

        result = await run(test_run_input)
        print("Run method result:")
        print(json.dumps(result, indent=2))
    
    asyncio.run(test_orchestrator()) 