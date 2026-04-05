"""
Creative reasoning functions extracted from WorldModel.

Handles creative composition queries including poem, story, and general
creative writing structure generation.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _creative_reasoning(wm, query: str, **kwargs) -> Dict[str, Any]:
    """
    Handle creative composition queries.

    Process:
    1. Identify creative task type (poem, story, essay)
    2. Analyze subject and requirements
    3. Generate creative structure (themes, form, imagery)
    4. Return structured output for OpenAI to translate into natural language
    """
    logger.info("[WorldModel] Creative reasoning engaged")
    query_lower = query.lower()

    # Detect creative task type
    if 'poem' in query_lower:
        task_type = 'poem'
    elif 'story' in query_lower:
        task_type = 'story'
    elif 'essay' in query_lower:
        task_type = 'essay'
    else:
        task_type = 'creative_writing'

    # Extract subject
    subject = _extract_creative_subject(wm, query)

    if task_type == 'poem':
        return _generate_poem_structure(wm, subject, query)
    elif task_type == 'story':
        return _generate_story_structure(wm, subject, query)
    else:
        return _generate_creative_structure(wm, subject, query)


def _extract_creative_subject(wm, query: str) -> str:
    """Extract the creative subject from query."""
    query_lower = query.lower()

    # Remove common prefixes
    prefixes = ['write a poem about', 'write a story about', 'write about',
               'poem about', 'story about', 'compose a poem about', 'create a']
    for prefix in prefixes:
        if prefix in query_lower:
            subject = query_lower.split(prefix)[1].strip()
            # Take first few words as subject
            words = subject.split()
            if words:
                return ' '.join(words[:3])

    # Fallback: look for common subjects
    subjects = ['cat', 'dog', 'ocean', 'mountain', 'love', 'time', 'nature', 'moon', 'sun']
    for subject in subjects:
        if subject in query_lower:
            return subject

    return 'the subject'


def _analyze_themes(wm, subject: str) -> list:
    """Analyze subject to determine appropriate themes."""
    subject_lower = subject.lower()

    theme_mappings = {
        'cat': ['independence', 'mystery', 'grace', 'nocturnal'],
        'dog': ['loyalty', 'companionship', 'joy', 'unconditional love'],
        'ocean': ['vastness', 'mystery', 'power', 'tranquility'],
        'mountain': ['strength', 'permanence', 'challenge', 'majesty'],
        'time': ['passage', 'change', 'memory', 'inevitability'],
        'love': ['connection', 'vulnerability', 'joy', 'loss'],
        'moon': ['mystery', 'cycles', 'reflection', 'solitude'],
        'sun': ['warmth', 'life', 'energy', 'hope'],
    }

    # Find matching themes
    for key, themes in theme_mappings.items():
        if key in subject_lower:
            return themes[:3]

    # Default themes
    return ['beauty', 'nature', 'observation']


def _determine_tone(wm, subject: str) -> str:
    """Determine appropriate tone for subject."""
    subject_lower = subject.lower()

    if any(word in subject_lower for word in ['cat', 'mystery', 'night', 'moon']):
        return 'mysterious_playful'
    elif any(word in subject_lower for word in ['dog', 'friend', 'joy', 'sun']):
        return 'warm_affectionate'
    elif any(word in subject_lower for word in ['ocean', 'mountain', 'sky']):
        return 'majestic_contemplative'
    else:
        return 'thoughtful_elegant'


def _select_imagery(wm, subject: str) -> list:
    """Select appropriate imagery categories."""
    subject_lower = subject.lower()

    imagery_maps = {
        'cat': ['shadows', 'moonlight', 'whiskers', 'velvet', 'silence'],
        'ocean': ['waves', 'foam', 'depths', 'horizon', 'salt'],
        'mountain': ['peaks', 'snow', 'stone', 'wind', 'majesty'],
        'moon': ['silver', 'glow', 'night', 'tides', 'phases'],
    }

    for key, imagery in imagery_maps.items():
        if key in subject_lower:
            return imagery

    return ['visual', 'tactile', 'movement']


def _generate_poem_structure(wm, subject: str, query: str) -> Dict[str, Any]:
    """Generate structured poem composition."""
    logger.info(f"[WorldModel] Generating poem structure for subject: {subject}")

    themes = _analyze_themes(wm, subject)
    tone = _determine_tone(wm, subject)
    imagery = _select_imagery(wm, subject)

    structure = {
        'type': 'poem',
        'subject': subject,
        'themes': themes,
        'form': {
            'stanzas': 4,
            'lines_per_stanza': 4,
            'rhyme_scheme': 'ABAB',
            'meter': 'flexible'
        },
        'literary_devices': ['metaphor', 'imagery', 'personification'],
        'tone': tone,
        'imagery_categories': imagery
    }

    # Build composition outline
    outline = []
    outline.append(f"Stanza 1: Introduce {subject} with primary imagery")
    if themes:
        outline.append(f"Stanza 2: Develop theme of {themes[0]}")
        if len(themes) > 1:
            outline.append(f"Stanza 3: Explore {themes[1]} through metaphor")
        if len(themes) > 2:
            outline.append(f"Stanza 4: Conclude with {themes[2]} and emotional resonance")

    response = f"""**VULCAN Creative Structure for Poem about {subject}:**

**Themes:** {', '.join(themes)}
**Form:** {structure['form']['stanzas']} stanzas, {structure['form']['rhyme_scheme']} rhyme scheme
**Tone:** {tone.replace('_', ' ')}
**Imagery:** {', '.join(imagery)}

**Composition Outline:**
{chr(10).join(outline)}

[This creative structure should be translated into flowing verse with the specified form and themes.]
"""

    return {
        'response': response,
        'confidence': 0.90,
        'reasoning_trace': structure,
        'mode': 'creative',
        'requires_llm_translation': True
    }


def _generate_story_structure(wm, subject: str, query: str) -> Dict[str, Any]:
    """Generate structured story composition."""
    logger.info(f"[WorldModel] Generating story structure for subject: {subject}")

    themes = _analyze_themes(wm, subject)

    structure = {
        'type': 'story',
        'subject': subject,
        'themes': themes,
        'structure': {
            'setting': f'A world where {subject} plays a central role',
            'protagonist': f'A character whose life intersects with {subject}',
            'conflict': f'A challenge or discovery related to {subject}',
            'resolution': f'Wisdom or transformation through {subject}'
        }
    }

    response = f"""**VULCAN Creative Structure for Story about {subject}:**

**Themes:** {', '.join(themes)}
**Setting:** {structure['structure']['setting']}
**Protagonist:** {structure['structure']['protagonist']}
**Conflict:** {structure['structure']['conflict']}
**Resolution:** {structure['structure']['resolution']}

**Story Arc:**
1. Introduction: Establish the world and protagonist
2. Rising Action: Introduce the central conflict
3. Climax: The pivotal moment of change
4. Resolution: The transformation and new understanding

[This creative structure should be translated into a compelling narrative.]
"""

    return {
        'response': response,
        'confidence': 0.85,
        'reasoning_trace': structure,
        'mode': 'creative',
        'requires_llm_translation': True
    }


def _generate_creative_structure(wm, subject: str, query: str) -> Dict[str, Any]:
    """Generate generic creative writing structure."""
    logger.info(f"[WorldModel] Generating creative structure for subject: {subject}")

    themes = _analyze_themes(wm, subject)

    response = f"""**VULCAN Creative Structure for writing about {subject}:**

**Themes:** {', '.join(themes)}
**Approach:** Thoughtful exploration of the subject
**Elements:** Description, reflection, insight

**Structure:**
1. Opening: Capture attention with vivid imagery
2. Development: Explore different aspects of {subject}
3. Reflection: Connect {subject} to broader themes
4. Conclusion: Leave the reader with lasting impression

[This creative structure should be translated into engaging prose.]
"""

    return {
        'response': response,
        'confidence': 0.80,
        'reasoning_trace': {
            'type': 'creative_writing',
            'subject': subject,
            'themes': themes
        },
        'mode': 'creative',
        'requires_llm_translation': True
    }
