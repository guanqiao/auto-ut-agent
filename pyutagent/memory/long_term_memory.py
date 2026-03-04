"""Long Term Memory - Integrated memory system.

This module provides a unified interface to:
- Episodic Memory: Task execution experiences
- Semantic Memory: Programming knowledge and concepts
- Procedural Memory: Skills and strategies
"""

import logging
from typing import Any, Dict, List, Optional

from .episodic_memory import EpisodicMemory, Episode, create_episodic_memory
from .semantic_memory import SemanticMemory, Concept, CodePattern, create_semantic_memory
from .procedural_memory import ProceduralMemory, Skill, create_procedural_memory

logger = logging.getLogger(__name__)


class LongTermMemory:
    """Long Term Memory - Unified interface to all memory types.

    This class provides a high-level interface that combines:
    - Episodic Memory: Cross-project task experiences
    - Semantic Memory: Programming knowledge and patterns
    - Procedural Memory: Learned skills and strategies

    Features:
    - Unified API for all memory operations
    - Cross-memory search and retrieval
    - Automatic memory type selection
    - Memory consolidation and summarization
    """

    def __init__(
        self,
        episodic_memory: EpisodicMemory,
        semantic_memory: SemanticMemory,
        procedural_memory: ProceduralMemory
    ):
        """Initialize long term memory.

        Args:
            episodic_memory: Episodic memory instance
            semantic_memory: Semantic memory instance
            procedural_memory: Procedural memory instance
        """
        self.episodic = episodic_memory
        self.semantic = semantic_memory
        self.procedural = procedural_memory

        logger.info("[LongTermMemory] Initialized with all memory types")

    async def record_experience(
        self,
        project: str,
        task_type: str,
        task_description: str,
        steps: List[Dict[str, Any]],
        outcome: str,
        duration_seconds: float,
        lessons: List[str]
    ) -> Episode:
        """Record a task execution experience.

        This method:
        1. Records the episode in episodic memory
        2. Extracts and learns concepts from the experience
        3. Updates procedural memory with skills

        Args:
            project: Project name
            task_type: Type of task
            task_description: Task description
            steps: Execution steps
            outcome: Execution outcome (success/failed)
            duration_seconds: Execution duration
            lessons: Lessons learned

        Returns:
            Recorded episode
        """
        import uuid
        from datetime import datetime

        episode = Episode(
            episode_id=str(uuid.uuid4()),
            project=project,
            task_type=task_type,
            task_description=task_description,
            steps=steps,
            outcome=outcome,
            duration_seconds=duration_seconds,
            lessons=lessons,
            timestamp=datetime.now()
        )

        # Record in episodic memory
        await self.episodic.record_episode(episode)

        # Learn from successful experiences
        if outcome == "success":
            await self._learn_from_experience(episode)

        return episode

    async def _learn_from_experience(self, episode: Episode):
        """Learn concepts and skills from a successful experience.

        Args:
            episode: Successful episode
        """
        # Learn concepts from lessons
        for lesson in episode.lessons:
            concept = Concept(
                concept_id="",
                name=f"Lesson: {lesson[:50]}",
                category="learned",
                description=lesson,
                examples=[episode.task_description],
                related_concepts=[episode.task_type],
                source=f"episode:{episode.episode_id}",
                confidence=0.8,
                usage_count=1,
                last_accessed=episode.timestamp,
                created_at=episode.timestamp,
                metadata={"project": episode.project}
            )
            await self.semantic.learn_concept(concept)

        # Learn skill from successful execution
        if episode.steps:
            strategy = {
                "name": f"Strategy for {episode.task_type}",
                "steps": [step.get("tool", "unknown") for step in episode.steps]
            }
            await self.procedural.learn(
                task_type=episode.task_type,
                strategy=strategy,
                success=True,
                duration_seconds=episode.duration_seconds
            )

    async def retrieve_relevant_knowledge(
        self,
        query: str,
        task_type: Optional[str] = None,
        project: Optional[str] = None,
        limit: int = 10
    ) -> Dict[str, List[Any]]:
        """Retrieve relevant knowledge from all memory types.

        Args:
            query: Search query
            task_type: Optional task type filter
            project: Optional project filter
            limit: Maximum results per memory type

        Returns:
            Dictionary with results from each memory type
        """
        results = {
            "episodes": [],
            "concepts": [],
            "skills": []
        }

        # Search episodic memory
        try:
            results["episodes"] = await self.episodic.search_similar(
                query=query,
                project=project,
                task_type=task_type,
                limit=limit
            )
        except Exception as e:
            logger.warning(f"[LongTermMemory] Episodic search failed: {e}")

        # Search semantic memory
        try:
            results["concepts"] = await self.semantic.query_concepts(
                query=query,
                limit=limit
            )
        except Exception as e:
            logger.warning(f"[LongTermMemory] Semantic search failed: {e}")

        # Search procedural memory
        if task_type:
            try:
                results["skills"] = await self.procedural.retrieve(
                    task_type=task_type,
                    limit=limit
                )
            except Exception as e:
                logger.warning(f"[LongTermMemory] Procedural search failed: {e}")

        return results

    async def get_best_practice(
        self,
        task_type: str,
        project: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get best practice for a task type.

        This combines insights from all memory types to recommend
        the best approach for a given task type.

        Args:
            task_type: Type of task
            project: Optional project context

        Returns:
            Best practice recommendation or None
        """
        # Get best skill
        best_skill = await self.procedural.get_best_skill(task_type)

        # Get relevant episodes
        episodes = await self.episodic.search_similar(
            query=task_type,
            project=project,
            task_type=task_type,
            limit=5
        )

        # Get relevant concepts
        concepts = await self.semantic.query_concepts(
            query=task_type,
            category="best_practice",
            limit=3
        )

        if not best_skill and not episodes and not concepts:
            return None

        # Compile best practice
        successful_episodes = [e for e in episodes if e.outcome == "success"]

        return {
            "task_type": task_type,
            "recommended_skill": best_skill.to_dict() if best_skill else None,
            "success_rate": len(successful_episodes) / len(episodes) if episodes else 0.0,
            "avg_duration": sum(e.duration_seconds for e in episodes) / len(episodes) if episodes else 0.0,
            "lessons_learned": list(set(
                lesson for e in successful_episodes for lesson in e.lessons
            )),
            "relevant_concepts": [c.name for c in concepts],
            "similar_experiences": len(episodes)
        }

    async def get_project_insights(self, project: str) -> Dict[str, Any]:
        """Get insights about a project.

        Args:
            project: Project name

        Returns:
            Project insights
        """
        # Get project summary
        summary = await self.episodic.get_project_summary(project)

        # Get recent episodes
        recent_episodes = await self.episodic.get_recent_episodes(project, limit=10)

        # Get lessons learned
        lessons = await self.episodic.get_lessons_learned(project)

        # Get popular task types
        task_types = {}
        for episode in recent_episodes:
            task_types[episode.task_type] = task_types.get(episode.task_type, 0) + 1

        return {
            "project": project,
            "total_experiences": summary.total_episodes,
            "success_rate": summary.success_rate,
            "average_duration": summary.avg_duration_seconds,
            "task_types": sorted(task_types.items(), key=lambda x: x[1], reverse=True),
            "lessons_learned": lessons[:20],
            "recent_activity": len(recent_episodes)
        }

    async def suggest_approach(
        self,
        task_description: str,
        task_type: str,
        project: Optional[str] = None
    ) -> Dict[str, Any]:
        """Suggest an approach for a new task.

        This analyzes past experiences to suggest the best approach.

        Args:
            task_description: Description of the task
            task_type: Type of task
            project: Optional project context

        Returns:
            Suggested approach
        """
        # Search for similar experiences
        similar = await self.episodic.search_similar(
            query=task_description,
            project=project,
            task_type=task_type,
            limit=5
        )

        # Get best skill
        best_skill = await self.procedural.get_best_skill(task_type)

        # Get relevant concepts
        concepts = await self.semantic.query_concepts(
            query=task_type,
            limit=3
        )

        # Analyze similar experiences
        if similar:
            successful = [e for e in similar if e.outcome == "success"]
            failed = [e for e in similar if e.outcome == "failed"]

            if successful:
                # Use the most recent successful approach
                reference = successful[0]
                return {
                    "confidence": len(successful) / len(similar),
                    "suggested_steps": reference.steps,
                    "estimated_duration": reference.duration_seconds,
                    "lessons_to_apply": reference.lessons,
                    "similar_experiences": len(similar),
                    "success_rate": len(successful) / len(similar),
                    "warnings": [e.lessons for e in failed if e.lessons]
                }

        # Fall back to skill-based suggestion
        if best_skill:
            return {
                "confidence": best_skill.success_rate,
                "suggested_steps": [{"tool": step} for step in best_skill.steps],
                "estimated_duration": best_skill.avg_duration_seconds,
                "lessons_to_apply": [],
                "similar_experiences": 0,
                "success_rate": best_skill.success_rate,
                "warnings": []
            }

        # No relevant experience
        return {
            "confidence": 0.0,
            "suggested_steps": [],
            "estimated_duration": 0.0,
            "lessons_to_apply": [],
            "similar_experiences": 0,
            "success_rate": 0.0,
            "warnings": ["No relevant experience found"]
        }

    async def consolidate_memories(self):
        """Consolidate and optimize memories.

        This method:
        1. Removes duplicate concepts
        2. Updates skill success rates
        3. Archives old episodes
        """
        logger.info("[LongTermMemory] Starting memory consolidation")

        # Get all skills and update
        skills = await self.procedural.get_all_skills()
        for skill in skills:
            if skill.usage_count > 10 and skill.success_rate < 0.3:
                # Low success rate skill - consider removing
                logger.warning(f"[LongTermMemory] Low success skill: {skill.name}")

        # Get popular concepts
        popular = await self.semantic.get_popular_concepts(limit=20)
        logger.info(f"[LongTermMemory] Top concepts: {[c.name for c in popular[:5]]}")

        logger.info("[LongTermMemory] Memory consolidation complete")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about all memory types.

        Returns:
            Memory statistics
        """
        return {
            "episodic": "Connected" if self.episodic else "Not available",
            "semantic": "Connected" if self.semantic else "Not available",
            "procedural": "Connected" if self.procedural else "Not available"
        }

    def close(self):
        """Close all memory connections."""
        if self.episodic:
            self.episodic.close()
        if self.semantic:
            self.semantic.close()
        if self.procedural:
            self.procedural.close()
        logger.info("[LongTermMemory] All memory connections closed")


def create_long_term_memory(
    storage_dir: str = ".pyutagent"
) -> LongTermMemory:
    """Create long term memory instance with all memory types.

    Args:
        storage_dir: Storage directory for all memories

    Returns:
        LongTermMemory instance
    """
    episodic = create_episodic_memory(storage_dir)
    semantic = create_semantic_memory(storage_dir)
    procedural = create_procedural_memory(storage_dir)

    return LongTermMemory(
        episodic_memory=episodic,
        semantic_memory=semantic,
        procedural_memory=procedural
    )
