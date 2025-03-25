import json
import re
from typing import List, Dict, Any
from naptha_sdk.schemas import KBRunInput, KBDeployment, NodeConfig
from naptha_sdk.modules.kb import KnowledgeBase
from naptha_sdk.inference import InferenceClient
from group_chat_orchestrator.utils import get_logger
from group_chat_orchestrator.prompts import create_summary_prompt

logger = get_logger(__name__)

def clean_text(text: str) -> str:
    """Clean and normalize text by removing all special characters and newlines."""
    if not text:
        return ""
    
    text = text.encode('utf-8').decode('unicode_escape')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    text = text.strip()
    return text

async def get_relevant_context(
    kb: KnowledgeBase,
    kb_deployment: KBDeployment,
    run_id: str,
    topic: str,
    consumer_id: str,
    signature: str
) -> Dict[str, Any]:
    """Get relevant context from the knowledge base.
    
    Args:
        kb: Knowledge base instance
        kb_deployment: Knowledge base deployment configuration
        run_id: Unique identifier for the run
        topic: Topic to get context for
        consumer_id: Consumer ID for authentication
        signature: Signature for authentication
        
    Returns:
        Dict containing previous questions and relevant findings
    """
    try:
        kb_result = await kb.run(KBRunInput(
            consumer_id=consumer_id,
            inputs={
                "func_name": "get_relevant_context",
                "func_input_data": {
                    "query": topic
                }
            },
            deployment=kb_deployment,
            signature=signature
        ))

        if isinstance(kb_result.results[-1], str):
            try:
                kb_data = json.loads(kb_result.results[-1])
            except json.JSONDecodeError:
                logger.error("Failed to parse KB response as JSON")
                return {"previous_questions": [], "relevant_findings": []}
        else:
            kb_data = kb_result.results[-1]

        findings = []
        if isinstance(kb_data, list):
            for item in kb_data:
                if isinstance(item, dict) and "findings" in item:
                    findings.extend(item["findings"])
        elif isinstance(kb_data, dict) and "findings" in kb_data:
            findings.extend(kb_data["findings"])

        seen_sections = set()
        unique_findings = []
        
        for finding in findings:
            section = finding.get('section', '').lower()
            points = finding.get('points', [])
            
            if any(section in seen or seen in section for seen in seen_sections):
                continue
                
            if len(points) < 2:
                continue
                
            if not any(len(point.strip()) > 50 for point in points):
                continue
                
            seen_sections.add(section)
            unique_findings.append(finding)

        questions = []
        if isinstance(kb_data, list):
            if len(kb_data) >= 2:
                prev_response = kb_data[-2]
                if isinstance(prev_response, dict) and "questions" in prev_response:
                    questions = prev_response["questions"]
        elif isinstance(kb_data, dict) and "questions" in kb_data:
            questions = kb_data["questions"]

        clean_questions = [clean_text(q) for q in questions]
        unique_questions = list(dict.fromkeys(clean_questions))
        
        if len(unique_questions) < 3:
            unique_questions.extend([
                "What are the key challenges and limitations in this area?",
                "How can these findings be applied in real-world scenarios?",
                "What future developments or trends should be considered?"
            ])

        logger.info(f"Previous questions for context: {json.dumps(unique_questions, indent=2)}")
        logger.info(f"Relevant findings for context: {json.dumps(unique_findings, indent=2)}")

        return {
            "previous_questions": unique_questions,
            "relevant_findings": unique_findings
        }

    except Exception as e:
        logger.error(f"Error getting relevant context: {str(e)}")

async def summarize_questions_to_topic(
    questions: List[str],
    topic: str,
    inference_client: InferenceClient,
    node_config: NodeConfig
) -> str:
    """Summarize a list of questions into a coherent topic for the next agent.
    
    Args:
        questions: List of questions to summarize
        topic: Original research topic
        agent: Agent instance to use for inference
        node_config: Node configuration for the agent
        
    Returns:
        str: Summarized topic
    """
    try:
        summary_prompt = create_summary_prompt(questions, topic)

        response = await inference_client.run_inference({
            "model": node_config.llm_config.model,
            "messages": [
                {"role": "system", "content": "You are a research assistant that creates focused, specific research topics."},
                {"role": "user", "content": summary_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 200
        })

        if isinstance(response, dict):
            topic = response['choices'][0]['message']['content']
        else:
            topic = response.choices[0].message.content

        return clean_text(topic)
    except Exception as e:
        logger.error(f"Error summarizing questions to topic: {str(e)}")
        return topic

def format_content_for_agent(
    research_topic: str,
    message_history: List[Dict[str, str]],
    selected_questions: List[str],
    context: Dict[str, Any] = None
) -> str:
    """Format and clean content for the next agent."""
    initial_summary = clean_text(context.get("summary", "")) if context else ""
    questions_to_answer = context.get("questions_to_answer", []) if context else []
    
    recent_findings = []
    for message in message_history[-2:]:
        try:
            content = json.loads(message["content"])
            findings = extract_findings_to_list(content.get("findings", []))
            relevant_findings = []
            for finding in findings:
                if any(q.lower() in finding.lower() for q in questions_to_answer):
                    relevant_findings.append(finding)
            recent_findings.extend(relevant_findings)
        except json.JSONDecodeError:
            logger.error("Failed to parse message content")
    
    recent_findings = [clean_text(f) for f in recent_findings]
    questions = [clean_text(q) for q in selected_questions]
    
    agent_role = "Critical Analyst" if context and context.get("round", 1) == 2 else "Practical Implementer"
    
    prompt = f"""Research Topic: {clean_text(research_topic)}

You are the {agent_role} in this research process. Your role is to:
{'- Challenge assumptions and provide balanced analysis' if agent_role == 'Critical Analyst' else '- Focus on practical applications and implementation'}

Your task is to focus specifically on answering these questions:
{chr(10).join(f"- {q}" for q in questions)}

Previous relevant findings that may help answer these questions:
{chr(10).join(f"- {f}" for f in recent_findings)}

Please provide:
1. Detailed findings that directly address the questions from your role's perspective
2. Focus on new insights and perspectives not covered in previous findings
3. Consider practical implications and real-world applications
4. Include specific examples or case studies where relevant
5. Address any potential challenges or limitations
6. Avoid repeating information from previous findings
7. Generate follow-up questions that explore new aspects

Remember to maintain your role's perspective and focus specifically on answering the given questions rather than providing a general overview."""
    
    return prompt

def process_agent_response(finding: Any) -> Dict[str, Any]:
    """Process and structure the agent's response."""
    try:
        if isinstance(finding, str):
            finding = json.loads(finding)
        
        questions_to_answer = finding.get('questions_to_answer', [])
        
        findings = finding.get('findings', [])
        if isinstance(findings, list):
            if all(isinstance(f, str) for f in findings):
                current_section = None
                current_points = []
                structured_findings = []
                
                for line in findings:
                    line = line.strip()
                    if not line:
                        continue
                        
                    if any(line.startswith(prefix) for prefix in [
                        'Section:', 'Findings for:', '###', '#', '**', 
                        '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.'
                    ]):
                        if current_section and current_points:
                            structured_findings.append({
                                "section": current_section,
                                "points": current_points
                            })
                        current_section = line
                        for prefix in ['Section:', 'Findings for:', '###', '#', '**']:
                            if current_section.startswith(prefix):
                                current_section = current_section[len(prefix):].strip()
                        if current_section[0].isdigit() and '. ' in current_section:
                            current_section = current_section.split('. ', 1)[1]
                        current_points = []
                    elif line.startswith('-') or line.startswith('*'):
                        point = line.lstrip('-* ').strip()
                        if point and current_section:
                            current_points.append(point)
                    elif current_section:
                        current_points.append(line)
                    elif not current_section and line:
                        current_section = line
                        current_points = []
                
                if current_section and current_points:
                    structured_findings.append({
                        "section": current_section,
                        "points": current_points
                    })
                
                if not structured_findings and questions_to_answer:
                    for question in questions_to_answer:
                        relevant_points = [
                            f for f in findings 
                            if question.lower() in f.lower() or any(word in f.lower() for word in question.lower().split())
                        ]
                        if relevant_points:
                            structured_findings.append({
                                "section": f"Findings for: {question}",
                                "points": relevant_points
                            })
                
                if not structured_findings:
                    structured_findings = [{
                        "section": "Key Findings",
                        "points": findings
                    }]
                
                findings = structured_findings
            
            elif all(isinstance(f, dict) for f in findings):
                for f in findings:
                    if "section" not in f:
                        f["section"] = "Key Findings"
                    if "points" not in f:
                        f["points"] = [str(f.get("content", ""))] if "content" in f else []
        
        questions = finding.get('questions', [])
        metadata = finding.get('metadata', {})
        
        cleaned_findings = []
        for finding_item in findings:
            if isinstance(finding_item, dict):
                section = clean_text(finding_item.get('section', ''))
                points = [clean_text(p) for p in finding_item.get('points', [])]
                points = [p for p in points if p.strip()]
                if points:
                    cleaned_findings.append({
                        "section": section,
                        "points": points
                    })
        
        cleaned_questions = [clean_text(q) for q in questions]
        cleaned_questions = [q for q in cleaned_questions if q.strip()]
        
        if "round" in metadata:
            metadata["role"] = "Initial Researcher" if metadata["round"] == 1 else \
                             "Critical Analyst" if metadata["round"] == 2 else \
                             "Practical Implementer"
        
        return {
            "findings": cleaned_findings,
            "questions": cleaned_questions,
            "metadata": metadata,
            "questions_to_answer": questions_to_answer
        }
    except json.JSONDecodeError:
        return {
            "findings": [{
                "section": "Key Findings",
                "points": [clean_text(str(finding))]
            }],
            "questions": [],
            "metadata": {},
            "questions_to_answer": []
        }

def extract_findings_to_list(findings: List[Any]) -> List[str]:
    """Extract findings into a flat list of strings."""
    all_findings = []
    if isinstance(findings, list):
        for finding_item in findings:
            if isinstance(finding_item, dict):
                section = finding_item.get('section', '')
                points = finding_item.get('points', [])
                role = finding_item.get('metadata', {}).get('role', '')
                if role:
                    all_findings.extend([f"{role} - {section}: {point}" for point in points if point.strip()])
                else:
                    all_findings.extend([f"{section}: {point}" for point in points if point.strip()])
            else:
                all_findings.append(str(finding_item))
    elif findings:
        all_findings.append(str(findings))
    return all_findings 