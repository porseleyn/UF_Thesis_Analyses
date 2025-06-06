def extract_indicators_teamB(self, df, game_id):
        # First call the compute function to populate instance variables
        self.compute_game_stats(df)
        # Now you can use them
        df_teamB_offense = self.df_teamB_offense
        b_total_minutes = self.b_total_minutes
        total_gametime_excl_breaks = self.total_gametime_excl_breaks
        b_intervals = self.b_intervals
        timeout_intervals = self.timeout_intervals
        
        indicators_B = {}

        key = f"{game_id}_teamB"
        
        #1
        scores_B = df_teamB_offense['Behaviour'].value_counts().get('Score', 0)
        att_scores_B = df_teamB_offense['Behaviour'].value_counts().get('Scoring_attempt', 0)
        f_turnover_B = df_teamB_offense[(df_teamB_offense['Behaviour'] == 'Forced_turnover') & (df_teamB_offense['Field_Loc'] == 'EZ_Off')].shape[0]
        uf_turnover_B = df_teamB_offense[(df_teamB_offense['Behaviour'] == 'Unforced_turnover') & (df_teamB_offense['Field_Loc'] == 'EZ_Off')].shape[0]
        score_attempts_B = scores_B + att_scores_B + f_turnover_B + uf_turnover_B
        
        #2
        score_efficiency_B = scores_B / score_attempts_B
        
        #3
        disc_possession_B = b_total_minutes / total_gametime_excl_breaks * 100

        #4
        succ_pass_B = df_teamB_offense['Behaviour'].value_counts().get('Successful_pass', 0) + scores_B
        f_turnover_B = df_teamB_offense['Behaviour'].value_counts().get('Forced_turnover', 0)
        uf_turnover_B = df_teamB_offense['Behaviour'].value_counts().get('Unforced_turnover', 0)
        total_pass_B = succ_pass_B + f_turnover_B + uf_turnover_B + att_scores_B
        pass_acc_B = succ_pass_B / total_pass_B * 100
        
        #7
        total_turnover_B = f_turnover_B + uf_turnover_B
        
        #10
        score_times = df_teamB_offense[df_teamB_offense['Behaviour'] == 'Score']['Time'].tolist()
        scoring_pos_B = self.get_scoring_possessions(b_intervals, score_times)
        passes_per_score_B = self.count_passes_in_possessions(scoring_pos_B, df_teamB_offense)
        avg_passes_per_score_B = np.mean(passes_per_score_B) if passes_per_score_B else 0
        
        #11
        pass_rate_B = total_pass_B / b_total_minutes
        
        #12
        teamB_durations = self.subtract_timeouts(b_intervals, timeout_intervals)
        avg_teamB_sec = np.mean(teamB_durations)
        
        indicators_B[key] = {
                "score_attempts": score_attempts_B,
                "score_efficiency": score_efficiency_B,
                "disc_possession": disc_possession_B,
                "pass_acc": pass_acc_B,
                "f_turnover": f_turnover_B,
                "uf_turnover": uf_turnover_B,
                "total_turnover": total_turnover_B,
                "subs": df_teamB_offense['Subs_A'[0]],
                "early_win": df_teamB_offense['Early win'[0]],
                "avg_passes_per_score": avg_passes_per_score_B,
                "pass_rate": pass_rate_B,
                "avg_poss": avg_teamB_sec,
        }

        return indicators_B