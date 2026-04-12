import numpy as np

class PerPromptStatTracker:
    def __init__(self, global_std=False):
        self.global_std = global_std
        # Change stats to store dict of lists: stats[prompt] -> list of rewards (scalar)
        # We need to support stats[prompt][reward_key] -> list
        self.stats = {}
        self.history_prompts = set()

    def update(self, prompts, rewards, exp=False):
        # Existing logic handles scalar rewards.
        # We can keep this for backward compatibility or when only 1 reward exists.
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        unique = np.unique(prompts)
        advantages = np.zeros_like(rewards)
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = []
            elif isinstance(self.stats[prompt], dict):
                 # If we converted to dict support, this might break if we mix usage. 
                 # Let's separate storage for multi-reward if needed, 
                 # OR assume this method is only for scalar 'avg' reward if used cleanly.
                 pass
            self.stats[prompt].extend(prompt_rewards)
            self.history_prompts.add(hash(prompt))  # Add hash of prompt to history_prompts
        for prompt in unique:
            self.stats[prompt] = np.stack(self.stats[prompt])
            prompt_rewards = rewards[prompts == prompt]  # Fix: Recalculate prompt_rewards for each prompt
            mean = np.mean(self.stats[prompt], axis=0, keepdims=True)
            if self.global_std:
                std = np.std(rewards, axis=0, keepdims=True) + 1e-4  # Use global std of all rewards
            else:
                std = np.std(self.stats[prompt], axis=0, keepdims=True) + 1e-4
            advantages[prompts == prompt] = (prompt_rewards - mean) / std
        return advantages

    def update_multireward(self, prompts, reward_dict, weights_dict):
        """
        GDPO-style update. 
        We need to track stats for EACH reward key individually.
        We'll use a separate storage `self.stats_multi` to avoid conflict with scalar `self.stats`.
        """
        if not hasattr(self, 'stats_multi'):
            self.stats_multi = {} # {prompt: {reward_key: []}}
            
        prompts = np.array(prompts)
        unique = np.unique(prompts)
        
        # 1. Update history for each reward
        for key, rewards in reward_dict.items():
            rewards = np.array(rewards, dtype=np.float64)
            for prompt in unique:
                prompt_rewards = rewards[prompts == prompt]
                if prompt not in self.stats_multi:
                    self.stats_multi[prompt] = {}
                if key not in self.stats_multi[prompt]:
                    self.stats_multi[prompt][key] = []
                self.stats_multi[prompt][key].extend(prompt_rewards)
                self.history_prompts.add(hash(prompt))

        # 2. Compute normalized advantages per reward
        combined_advantages = np.zeros(len(prompts))
        
        for key, rewards in reward_dict.items():
            rewards = np.array(rewards, dtype=np.float64)
            adv = np.zeros_like(rewards)
            weight = weights_dict.get(key, 0.0)
            
            for prompt in unique:
                # Convert list to stack for calc
                history = np.stack(self.stats_multi[prompt][key])
               
                
                prompt_subset = rewards[prompts == prompt]
                mean = np.mean(history, axis=0, keepdims=True)
                
                # Vanilla GDPO uses group-wise std (std of history/group), effectively ignoring global_std if set.
                std = np.std(history, axis=0, keepdims=True) + 1e-4
                
                adv[prompts == prompt] = (prompt_subset - mean) / std
            
            combined_advantages += adv * weight
            
        # 3. Final Batch Normalization (GDPO Paper Step)
        # "advantages = (pre_bn_advantages - bn_advantages_mean) / (bn_advantages_std + 1e-4)"
        # where pre_bn_advantages is the weighted sum.
        
        return (combined_advantages - combined_advantages.mean()) / (combined_advantages.std() + 1e-4)

    def get_stats(self):
        if self.stats:
            avg_group_size = sum(len(v) for v in self.stats.values()) / len(self.stats)
        elif hasattr(self, 'stats_multi') and self.stats_multi:
            # stats_multi: {prompt: {key: [rewards...]}}
            total_size = 0
            count = 0
            for p_dict in self.stats_multi.values():
                if p_dict:
                    first_key = next(iter(p_dict))
                    total_size += len(p_dict[first_key])
                    count += 1
            avg_group_size = total_size / count if count > 0 else 0
        else:
            avg_group_size = 0
            
        history_prompts = len(self.history_prompts)
        return avg_group_size, history_prompts

    def clear(self):
        self.stats = {}
        if hasattr(self, 'stats_multi'):
            self.stats_multi = {}

    def get_mean_of_top_rewards(self, top_percentage):
        # GDPO support: aggregate from stats_multi if stats is empty
        stats_values = self.stats.values()
        if not stats_values and hasattr(self, 'stats_multi') and self.stats_multi:
            # Gather all rewards from all keys? Or just mean per prompt?
            # Existing logic gathers all scalars.
            # self.stats_multi[prompt][key] is a list of rewards.
            # We can flatten everything.
            all_rewards = []
            for p_dict in self.stats_multi.values():
                for r_list in p_dict.values():
                    all_rewards.extend(r_list)
            
            # To match structure (list of lists or similar), let's just make it a single list of values
            if not all_rewards:
                return 0.0
            
            rewards = np.array(all_rewards)
            if top_percentage == 100:
                return float(np.mean(rewards))
            
            threshold = np.percentile(rewards, 100 - top_percentage)
            return float(np.mean(rewards[rewards >= threshold]))

        if not stats_values:
            return 0.0

        assert 0 <= top_percentage <= 100

        per_prompt_top_means = []
        for prompt_rewards in stats_values:
            rewards = np.array(prompt_rewards) if isinstance(prompt_rewards, list) else prompt_rewards

            if rewards.size == 0:
                continue

            if top_percentage == 100:
                per_prompt_top_means.append(np.mean(rewards))
                continue

            lower_bound_percentile = 100 - top_percentage
            threshold = np.percentile(rewards, lower_bound_percentile)

            top_rewards = rewards[rewards >= threshold]

            if top_rewards.size > 0:
                per_prompt_top_means.append(np.mean(top_rewards))

        return float(np.mean(per_prompt_top_means)) if per_prompt_top_means else 0.0

class PerPromptAndPreferenceStatTracker:
    """
    Stat tracker for GDPO-style multi-reward normalization.
    
    Key design: Uses CURRENT BATCH statistics only for normalization.
    This is faithful to DiffusionNFT's online approach and avoids distribution shift.
    
    The 'group' is the K repeats of the same prompt in the current batch.
    No history accumulation - each batch is self-contained.
    """
    
    def __init__(self, global_std=False):
        self.global_std = global_std
        # Track unique prompts seen for logging only (not used for normalization)
        self.history_prompts = set()
        # Store current batch stats for logging
        self._last_group_sizes = []
        # Store current batch objective values for get_mean_of_top_rewards
        self._current_batch_objectives = None

    def update_from_reward_vectors_gdpo(self, prompts, reward_vec, preferences, weights, use_per_objective_loss=False):
        """
        GDPO: Normalize each reward channel using CURRENT BATCH statistics only.
        The 'group' is the K repeats of the same prompt in THIS batch.
        
        This is faithful to DiffusionNFT's online approach:
        - No history accumulation
        - No distribution shift from old samples
        - Clean baseline from current model capability
        
        Args:
            prompts: List of prompt strings (N,)
            reward_vec: (N, R) reward values for each objective
            preferences: (N, R) preference weights for each sample
            weights: (R,) global weights for each objective
        
        Returns:
            final_adv: (N,) normalized advantages
            objective: (N,) raw weighted objective for logging
        """
        prompts = np.array(prompts, dtype=object)
        reward_vec = np.asarray(reward_vec, dtype=np.float64)  # (N, R)
        preferences = np.asarray(preferences, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        
        N, R = reward_vec.shape
        unique_prompts = np.unique(prompts)
        advantages_per_channel = np.zeros_like(reward_vec)
        
        # Track group sizes for logging
        self._last_group_sizes = []
        
        # 1. Per-channel normalization using CURRENT BATCH group statistics
        for prompt in unique_prompts:
            mask = (prompts == prompt)
            group_rewards = reward_vec[mask]  # (K, R) - the K repeats for this prompt
            group_size = group_rewards.shape[0]
            self._last_group_sizes.append(group_size)
            
            for r in range(R):
                channel_rewards = group_rewards[:, r]  # (K,)
                mean = np.mean(channel_rewards)
                std = np.std(channel_rewards) + 1e-4
                advantages_per_channel[mask, r] = (reward_vec[mask, r] - mean) / std
            
            # Track for logging only
            self.history_prompts.add(hash(prompt))
        
        # Per-objective mode: return per-prompt normalized (N, R) advantages directly.
        # Skip extra batch normalization — per-prompt normalization already handles
        # scale differences, and re-normalizing per-channel destroys the natural
        # relative scaling that preferences need to create ordered Pareto fronts.
        if use_per_objective_loss:
            objective = (reward_vec * weights[None, :]).sum(axis=-1)
            self._current_batch_objectives = objective
            return advantages_per_channel, objective

        # 2. Scalarize: weighted sum using each sample's preference
        combined_advantages = (preferences * advantages_per_channel * weights[None, :]).sum(axis=-1)

        # 3. Final batch normalization (standard GRPO)
        final_adv = (combined_advantages - combined_advantages.mean()) / (combined_advantages.std() + 1e-4)

        # Compute objective for logging
        objective = (preferences * reward_vec * weights[None, :]).sum(axis=-1)

        # Store for get_mean_of_top_rewards
        self._current_batch_objectives = objective

        return final_adv, objective

    def get_stats(self):
        """Return average group size and number of unique prompts seen."""
        if self._last_group_sizes:
            avg_group_size = np.mean(self._last_group_sizes)
        else:
            avg_group_size = 0
        
        history_prompts = len(self.history_prompts)
        return avg_group_size, history_prompts

    def clear(self):
        """Clear batch-specific data. Called at end of each epoch."""
        self._last_group_sizes = []
        self._current_batch_objectives = None
        # Note: history_prompts is kept for cumulative logging

    def get_mean_of_top_rewards(self, top_percentage):
        """Get mean of top X% of objectives from current batch."""
        if self._current_batch_objectives is None or len(self._current_batch_objectives) == 0:
            return 0.0
        
        assert 0 <= top_percentage <= 100
        
        objectives = np.array(self._current_batch_objectives)
        
        if top_percentage == 100:
            return float(np.mean(objectives))
        
        threshold = np.percentile(objectives, 100 - top_percentage)
        top_objectives = objectives[objectives >= threshold]
        
        return float(np.mean(top_objectives)) if len(top_objectives) > 0 else 0.0

def main():
    tracker = PerPromptStatTracker()
    prompts = ["a", "b", "a", "c", "b", "a"]
    rewards = [1, 2, 3, 4, 5, 6]
    advantages = tracker.update(prompts, rewards)
    print("Advantages:", advantages)
    avg_group_size, history_prompts = tracker.get_stats()
    print("Average Group Size:", avg_group_size)
    print("History Prompts:", history_prompts)
    tracker.clear()
    print("Stats after clear:", tracker.stats)


if __name__ == "__main__":
    main()