"""
Curator: Integrates insights into the playbook.

The Curator synthesizes reflections into structured delta updates
and manages playbook evolution.
"""

from typing import Dict, List, Optional, Any
import json
from src.utils.llm_interface import LLMInterface
from src.ace.playbook import Playbook


class Curator:
    """
    Curator component of ACE framework.
    
    Integrates reflection insights into playbook through
    incremental delta updates.
    """
    
    def __init__(self, model: Optional[str] = None, temperature: float = 0.3):
        """
        Initialize Curator.
        
        Args:
            model: LLM model for curation (defaults to Ollama if not provided)
            temperature: Sampling temperature
        """
        self.llm = LLMInterface(
            model=model,
            temperature=temperature,
            max_tokens=2048
        )
    
    def curate(self, insights: List[Dict[str, Any]],
               current_playbook: Playbook,
               task_context: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate delta updates from insights.
        
        Args:
            insights: List of reflection outputs
            current_playbook: Current playbook state
            task_context: Optional context about the task
            
        Returns:
            List of delta bullet dictionaries to add to playbook
        """
        # Build curation prompt
        prompt = self._build_curation_prompt(
            insights=insights,
            current_playbook=current_playbook,
            task_context=task_context
        )
        
        # Generate curation response
        response = self.llm.generate(prompt)
        
        # Parse delta updates
        delta_bullets = self._parse_curation_response(response)
        
        return delta_bullets
    
    def _build_curation_prompt(self, insights: List[Dict[str, Any]],
                               current_playbook: Playbook,
                               task_context: Optional[str]) -> str:
        """Build curation prompt."""
        prompt_parts = [
            "You are a master curator of knowledge.",
            "\nYour job is to:",
            "1. Review reflections from recent task attempts",
            "2. Identify NEW insights missing from current playbook",
            "3. Avoid redundancy with existing content",
            "4. Create specific, actionable bullet points",
            "5. Organize insights into appropriate sections\n"
        ]
        
        # Add task context
        if task_context:
            prompt_parts.append(f"\n## Task Context\n{task_context}\n")
        
        # Add current playbook stats
        stats = current_playbook.get_stats()
        prompt_parts.append(f"\n## Current Playbook Statistics")
        prompt_parts.append(f"Total bullets: {stats['total_bullets']}")
        prompt_parts.append(f"Section distribution: {stats['section_counts']}\n")
        
        # Add sample of current playbook
        playbook_sample = current_playbook.to_prompt(max_bullets_per_section=5)
        prompt_parts.append(f"\n## Current Playbook (Sample)\n{playbook_sample}\n")
        
        # Add insights
        prompt_parts.append("\n## Recent Insights to Curate\n")
        for i, insight in enumerate(insights, 1):
            prompt_parts.append(f"\n### Insight {i}")
            prompt_parts.append(f"Key lessons: {insight.get('key_insights', [])}")
            prompt_parts.append(f"Root cause: {insight.get('root_cause_analysis', 'N/A')}")
            prompt_parts.append(f"Correction: {insight.get('correct_approach', 'N/A')}\n")
        
        # Output format
        prompt_parts.append(
            "\n## Required Output Format (JSON):\n"
            "{\n"
            '  "reasoning": "your analysis of what to add",\n'
            '  "operations": [\n'
            '    {\n'
            '      "type": "ADD",\n'
            '      "section": "section_name",\n'
            '      "content": "the new bullet content"\n'
            '    },\n'
            '    ...\n'
            '  ]\n'
            "}\n\n"
            f"Available sections: {', '.join(current_playbook.sections)}\n"
            "\nNote: Only include NEW insights not already in the playbook. "
            "Be specific and actionable."
        )
        
        return "\n".join(prompt_parts)
    
    def _parse_curation_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse curation response into delta bullets."""
        try:
            parsed = json.loads(response)
            operations = parsed.get('operations', [])
            
            delta_bullets = []
            for op in operations:
                if op.get('type') == 'ADD':
                    delta_bullets.append({
                        'section': op.get('section'),
                        'content': op.get('content'),
                        'helpful_count': 0,
                        'harmful_count': 0,
                        'neutral_count': 0
                    })
            
            return delta_bullets
            
        except json.JSONDecodeError:
            # Fallback: try to extract bullet points from text
            return self._fallback_parse(response)
    
    def _fallback_parse(self, response: str) -> List[Dict[str, Any]]:
        """Fallback parsing for non-JSON responses."""
        delta_bullets = []
        lines = response.strip().split('\n')
        
        current_section = "strategies_and_hard_rules"  # Default section
        
        for line in lines:
            line = line.strip()
            
            # Check for section headers
            for section in Playbook.DEFAULT_SECTIONS:
                if section.replace('_', ' ').lower() in line.lower():
                    current_section = section
                    break
            
            # Check for bullet points
            if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                content = line.lstrip('-•* ').strip()
                if len(content) > 10:  # Only meaningful bullets
                    delta_bullets.append({
                        'section': current_section,
                        'content': content,
                        'helpful_count': 0,
                        'harmful_count': 0,
                        'neutral_count': 0
                    })
        
        return delta_bullets
    
    def apply_bullet_feedback(self, playbook: Playbook,
                             bullet_feedback: Dict[str, str]):
        """
        Apply feedback tags to bullets in playbook.
        
        Args:
            playbook: The playbook to update
            bullet_feedback: Dict mapping bullet_id to 'helpful'/'harmful'/'neutral'
        """
        for bullet_id, feedback in bullet_feedback.items():
            if bullet_id in playbook.bullets:
                playbook.update_bullet_feedback(bullet_id, feedback)
    
    def batch_curate(self, reflection_batches: List[List[Dict[str, Any]]],
                    playbook: Playbook,
                    deduplicate: bool = True) -> List[Dict[str, Any]]:
        """
        Curate insights from multiple batches.
        
        Args:
            reflection_batches: List of reflection lists (one per batch)
            playbook: Current playbook
            deduplicate: Whether to deduplicate after curation
            
        Returns:
            All delta bullets generated
        """
        all_deltas = []
        
        for batch in reflection_batches:
            deltas = self.curate(
                insights=batch,
                current_playbook=playbook
            )
            
            # Apply deltas to playbook
            playbook.update(deltas)
            
            # Apply bullet feedback
            for reflection in batch:
                if 'bullet_feedback' in reflection:
                    self.apply_bullet_feedback(
                        playbook,
                        reflection['bullet_feedback']
                    )
            
            all_deltas.extend(deltas)
        
        # Deduplicate if requested
        if deduplicate:
            playbook.deduplicate()
        
        return all_deltas
    
    def prune_harmful_bullets(self, playbook: Playbook, threshold: float = 0.5):
        """
        Remove bullets with high harmful-to-helpful ratio.
        
        Args:
            playbook: Playbook to prune
            threshold: Harmful ratio threshold for removal
        """
        bullets_to_remove = []
        
        for bullet_id, bullet in playbook.bullets.items():
            total_feedback = bullet.helpful_count + bullet.harmful_count
            
            if total_feedback > 5:  # Require minimum feedback
                harmful_ratio = bullet.harmful_count / total_feedback
                
                if harmful_ratio > threshold:
                    bullets_to_remove.append(bullet_id)
        
        # Remove harmful bullets
        for bullet_id in bullets_to_remove:
            bullet = playbook.bullets[bullet_id]
            playbook.section_bullets[bullet.section].remove(bullet_id)
            del playbook.bullets[bullet_id]
        
        return len(bullets_to_remove)
