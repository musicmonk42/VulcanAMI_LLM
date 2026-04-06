"""
Ethical principles extraction and analysis functions extracted from WorldModel.

Handles parsing dilemma structure, extracting moral principles, and analyzing
options against those principles.
"""

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _parse_dilemma_structure(wm, query: str) -> Dict[str, Any]:
    """
    Extract options and consequences from dilemma query.

    Industry Standard: Robust parsing with multiple fallback strategies.

    Args:
        query: The query containing the dilemma

    Returns:
        Dictionary with 'options' list containing option descriptions
    """
    try:
        query_lower = query.lower()
        structure = {'options': [], 'consequences': {}}

        # Strategy 1: Explicit option markers (A., B., Option A, Option B)
        if 'option a' in query_lower or 'a.' in query_lower:
            # Extract text after "A." or "option a" until next option or end
            a_match = re.search(r'(?:option\s+)?a[.:]\s*([^\n]+)', query, re.IGNORECASE)
            if a_match:
                structure['options'].append({
                    'id': 'A',
                    'description': a_match.group(1).strip()
                })

        if 'option b' in query_lower or 'b.' in query_lower:
            b_match = re.search(r'(?:option\s+)?b[.:]\s*([^\n]+)', query, re.IGNORECASE)
            if b_match:
                structure['options'].append({
                    'id': 'B',
                    'description': b_match.group(1).strip()
                })

        # Strategy 2: Trolley problem specific (pull lever / don't pull)
        if not structure['options'] and 'lever' in query_lower:
            structure['options'] = [
                {'id': 'A', 'description': 'Pull the lever'},
                {'id': 'B', 'description': 'Do not pull the lever'}
            ]

        # Strategy 3: Generic action/inaction
        if not structure['options']:
            structure['options'] = [
                {'id': 'A', 'description': 'Take action'},
                {'id': 'B', 'description': 'Do not take action'}
            ]

        # Extract consequences (numbers of people affected)
        numbers = re.findall(r'\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:people|person)', query_lower)
        structure['consequences']['numbers'] = numbers

        return structure

    except Exception as e:
        logger.warning(f"[WorldModel] Dilemma structure parsing error: {e}")
        return {'options': [], 'consequences': {}}


def _extract_moral_principles(wm, query: str) -> List[Dict[str, Any]]:
    """
    Extract moral principles mentioned in the query.

    Industry Standard: Comprehensive principle taxonomy with definitions.

    Args:
        query: The query text

    Returns:
        List of dictionaries with 'name' and 'description' of each principle
    """
    try:
        query_lower = query.lower()
        principles = []

        # Define principle patterns with their ethical meanings
        principle_patterns = {
            'non-instrumentalization': {
                'keywords': ['non-instrumentalization', 'means to an end', 'instrument', 'using people'],
                'description': 'People should not be used merely as means to an end'
            },
            'non-negligence': {
                'keywords': ['non-negligence', 'neglect', 'inaction', 'preventable', 'knowingly allow'],
                'description': 'It is impermissible to knowingly allow preventable harm through inaction'
            },
            'non-maleficence': {
                'keywords': ['do no harm', 'non-maleficence', 'harm', 'hurt', 'injure'],
                'description': 'One ought not to inflict harm on others'
            },
            'beneficence': {
                'keywords': ['beneficence', 'help', 'benefit', 'save', 'protect'],
                'description': 'One ought to prevent or remove harm and promote good'
            },
            'autonomy': {
                'keywords': ['autonomy', 'consent', 'choice', 'self-determination'],
                'description': 'Respect for persons and their right to make their own choices'
            },
            'justice': {
                'keywords': ['justice', 'fairness', 'equal', 'impartial'],
                'description': 'Fair and equitable treatment of all persons'
            }
        }

        # Check query against each principle pattern
        for principle_name, data in principle_patterns.items():
            if any(keyword in query_lower for keyword in data['keywords']):
                principles.append({
                    'name': principle_name,
                    'description': data['description']
                })

        # If no explicit principles found, infer from context
        if not principles:
            # Trolley problem implies these principles
            if 'trolley' in query_lower or 'save' in query_lower:
                principles.append({
                    'name': 'beneficence',
                    'description': 'One ought to prevent or remove harm'
                })
                principles.append({
                    'name': 'non-maleficence',
                    'description': 'One ought not to inflict harm'
                })

        logger.debug(f"[WorldModel] Extracted principles: {[p['name'] for p in principles]}")
        return principles

    except Exception as e:
        logger.warning(f"[WorldModel] Principle extraction error: {e}")
        return []


def _analyze_options_against_principles(
    wm,
    options: List[Dict[str, Any]],
    principles: List[Dict[str, Any]],
    query: str
) -> List[Dict[str, Any]]:
    """
    Analyze how each option relates to each moral principle.

    Industry Standard: Systematic analysis with clear compliance assessment.

    Args:
        options: List of option dictionaries from parse_dilemma_structure
        principles: List of principle dictionaries from extract_moral_principles
        query: Original query for context

    Returns:
        List of analysis dictionaries with considerations for each option
    """
    try:
        analysis = []
        query_lower = query.lower()

        for option in options:
            option_id = option.get('id', 'Unknown')
            option_desc = option.get('description', '')
            option_lower = option_desc.lower()

            # Analyze this option against each principle
            compliances = []
            violations = []

            for principle in principles:
                principle_name = principle['name']

                # Check compliance based on option type and principle
                if principle_name == 'non-instrumentalization':
                    # Pulling lever may use one person as means to save five
                    if 'pull' in option_lower and 'five' in query_lower:
                        violations.append(f"May violate {principle_name} by using one as means")
                    elif 'not' in option_lower or 'don\'t' in option_lower:
                        compliances.append(f"Avoids violating {principle_name}")

                elif principle_name == 'non-negligence':
                    # Inaction may violate non-negligence
                    if 'not' in option_lower or 'don\'t' in option_lower:
                        if 'five' in query_lower:
                            violations.append(f"Violates {principle_name} by allowing five deaths")
                    else:
                        compliances.append(f"Complies with {principle_name} by acting")

                elif principle_name == 'beneficence':
                    # Acting to save more people shows beneficence
                    if 'pull' in option_lower or 'take action' in option_lower:
                        compliances.append(f"Shows {principle_name} by saving lives")

                elif principle_name == 'non-maleficence':
                    # Any action that causes death may violate non-maleficence
                    if 'pull' in option_lower:
                        violations.append(f"May violate {principle_name} by causing one death")

            # Construct consideration text
            consideration = f"Option {option_id}: {option_desc}"
            if compliances:
                consideration += f" - Complies: {', '.join(compliances[:2])}"
            if violations:
                consideration += f" - Conflicts: {', '.join(violations[:2])}"

            analysis.append({
                'option': option_id,
                'description': option_desc,
                'compliances': compliances,
                'violations': violations,
                'consideration': consideration
            })

        return analysis

    except Exception as e:
        logger.warning(f"[WorldModel] Option analysis error: {e}")
        return []
