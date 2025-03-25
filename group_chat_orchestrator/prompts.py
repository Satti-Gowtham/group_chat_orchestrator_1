from typing import List, Dict, Any
from naptha_sdk.utils import get_logger

logger = get_logger(__name__)

def create_research_prompt(topic: str) -> str:
    """Create the initial research prompt for the researcher agent."""
    return (
        f"Research Topic: {topic}\n\n"
        "You are the Research Agent. Your role is to:\n"
        "- Conduct comprehensive initial research\n"
        "- Identify key areas and trends\n"
        "- Generate focused questions\n"
        "- Provide detailed findings with evidence\n\n"
        "Your response should include:\n"
        "- 7-8 distinct sections\n"
        "- 4-6 detailed points per section\n"
        "- Specific examples and evidence\n"
        "- 4-6 follow-up questions\n\n"
        "IMPORTANT: Focus on gathering factual information and evidence. "
        "Do not include analysis or synthesis - that will be done by other agents.\n\n"
        "Format each section as:\n"
        "Section: [Topic]\n"
        "- [Detailed point with evidence]\n"
        "- [Detailed point with evidence]\n"
        "etc."
    )

def create_agent_prompt(
    current_topic: str,
    summarised_topic: str,
    agent_name: str,
    context: Dict[str, Any]
) -> str:
    """Create the prompt for analyst and synthesizer agents."""
    if agent_name.lower() == "analyst":
        role_guidelines = (
            "You are the Critical Analyst. Your role is to:\n"
            "- Analyze previous findings critically\n"
            "- Challenge assumptions\n"
            "- Evaluate evidence quality\n"
            "- Identify gaps and biases\n"
            "- Consider ethical implications\n\n"
            "IMPORTANT: Do not repeat the basic findings. Instead:\n"
            "- Evaluate the strength of evidence for each finding\n"
            "- Identify potential biases or limitations\n"
            "- Challenge assumptions and explore alternative explanations\n"
            "- Consider ethical implications and potential harms\n"
            "- Suggest areas where more research is needed\n\n"
            "Your analysis should focus on:\n"
            "1. Evidence Quality: Evaluate the strength and reliability of evidence\n"
            "2. Methodological Limitations: Identify study design issues and biases\n"
            "3. Alternative Explanations: Consider other factors that might explain findings\n"
            "4. Ethical Considerations: Examine potential harms and benefits\n"
            "5. Research Gaps: Identify areas needing further investigation\n"
            "6. Implementation Challenges: Analyze practical barriers to applying findings\n"
            "7. Future Implications: Consider long-term consequences and trends\n\n"
            "Format your analysis as:\n"
            "Analysis Section: [Focus Area]\n"
            "- [Critical evaluation point with evidence]\n"
            "- [Limitation or alternative explanation]\n"
            "- [Implications or recommendations]"
        )
    elif agent_name.lower() == "synthesizer":
        role_guidelines = (
            "You are the Synthesizer. Your role is to:\n"
            "- Integrate all previous findings\n"
            "- Identify patterns and themes\n"
            "- Draw conclusions\n"
            "- Propose applications\n"
            "- Consider future implications\n\n"
            "IMPORTANT: Do not repeat the basic findings. Instead:\n"
            "- Connect findings across different areas\n"
            "- Identify emerging patterns and themes\n"
            "- Draw practical conclusions\n"
            "- Suggest real-world applications\n"
            "- Consider future implications and trends\n\n"
            "Your synthesis should focus on:\n"
            "1. Pattern Recognition: Identify common themes across findings\n"
            "2. Cross-Domain Connections: Link findings from different areas\n"
            "3. Practical Applications: Suggest real-world implementations\n"
            "4. Policy Implications: Consider regulatory and policy needs\n"
            "5. Future Scenarios: Project potential outcomes and trends\n"
            "6. Stakeholder Impact: Analyze effects on different groups\n"
            "7. Action Recommendations: Provide specific next steps\n\n"
            "Format your synthesis as:\n"
            "Synthesis Section: [Theme/Pattern]\n"
            "- [Connection or pattern identified]\n"
            "- [Practical application or implication]\n"
            "- [Recommendation or next step]"
        )
    else:
        role_guidelines = (
            "You are the Research Agent. Your role is to:\n"
            "- Provide detailed analysis\n"
            "- Focus on practical implications\n"
            "- Consider challenges\n"
            "- Generate new perspectives"
        )

    previous_findings = context.get('relevant_findings', [])
    if previous_findings:
        previous_findings_text = ""
        for finding in previous_findings:
            previous_findings_text += f"- {finding['section']}: {', '.join(finding['points'])}\n"
        previous_findings_text += "\n"

    previous_questions = context.get('previous_questions', [])
    previous_questions_text = ""
    if previous_questions:
        previous_questions_text = "Questions to address:\n"
        for q in previous_questions:
            previous_questions_text += f"- {q}\n"
        previous_questions_text += "\n"

    return (
        f"Topic: {summarised_topic}\n\n"
        f"{role_guidelines}\n\n"
        f"Remember to build upon {previous_findings_text} rather than repeating them."
    )

def create_summary_prompt(questions: List[str], topic: str) -> str:
    """Create the prompt for summarizing questions into a topic."""
    return (
        f"Based on the following questions about {topic}, "
        "create a focused research topic that encompasses all these questions. "
        "The topic should be specific, actionable, and guide the next phase of research.\n\n"
        "Questions to consider:\n"
        f"{chr(10).join(f'- {q}' for q in questions)}\n\n"
        "Provide a single, well-formed topic that captures the essence of these questions."
    )

def format_findings_for_prompt(findings: List[Dict[str, Any]]) -> str:
    """Format findings for inclusion in prompts."""
    return chr(10).join(f"- {finding['section']}: {', '.join(finding['points'])}" for finding in findings)

def format_questions_for_prompt(questions: List[str]) -> str:
    """Format questions for inclusion in prompts."""
    return chr(10).join(f"- {q}" for q in questions) 