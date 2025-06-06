import numpy as np
import os
import pandas as pd

class DataBatchCleaner:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.summary_data = []

    def clean_data(self, df):
        df = df.drop(columns=[
            'Observation date', 'Observation duration', 'Description', 'Observation type',
            'Source', 'Time offset (s)', 'Media duration (s)', 'FPS', 'Subject',
            'Behavioral category', 'Media file name', 'Image index', 'Image file path'
        ])
        df = df.rename(columns={
            "Behavior": "Behaviour",
            "Modifier #1": "Field_Loc",
            "Behavior type": "B_Type"
        })
        
        return df
    
    def subtract_intervals(self, base_intervals, sub_intervals):
        result = []
        for start, stop in base_intervals:
            temp = [(start, stop)]
            for sub_start, sub_stop in sub_intervals:
                new_temp = []
                for s, e in temp:
                    # Skip non-overlapping
                    if sub_stop <= s or sub_start >= e:
                        new_temp.append((s, e))
                    else:
                        # Chop out the timeout
                        if sub_start > s:
                            new_temp.append((s, sub_start))
                        if sub_stop < e:
                            new_temp.append((sub_stop, e))
                temp = new_temp
            result.extend(temp)
        return result
    
    def rows_in_intervals(self, df, intervals):
        return df[df['Time'].apply(lambda t: any(start <= t <= end for start, end in intervals))]
    
    def subtract_timeouts(self, possessions, timeouts):
        clean_durations = []
        for start, stop in possessions:
            duration = stop - start
            for t_start, t_stop in timeouts:
                overlap_start = max(start, t_start)
                overlap_end = min(stop, t_stop)
                if overlap_start < overlap_end:
                    duration -= (overlap_end - overlap_start)
            clean_durations.append(duration)
        return clean_durations
    
    def get_scoring_possessions(self, possessions, score_events):
        scoring_possessions = []
        for start, stop in possessions:
            if any((score_time >= start) and (score_time <= stop) for score_time in score_events):
                scoring_possessions.append((start, stop))
        return scoring_possessions
    
    def count_passes_in_possessions(self, possessions, team_df):
        pass_counts = []
        for start, stop in possessions:
            passes = team_df[
                (team_df['Time'] >= start) &
                (team_df['Time'] <= stop) &
                (team_df['Behaviour'] == 'Successful_pass')
            ]
            pass_counts.append(len(passes))
        return pass_counts
    
    def compute_game_stats(self, df):
        """Game stats about duration of the games, times on offense for each team."""
        game_start = df['Time'].min()
        game_end = df['Time'].max()
        total_gametime = (game_end - game_start) / 60
        
        first_half_start = df[(df['Behaviour'] == 'First half') & (df['B_Type'] == 'START')]['Time'].iloc[0]
        first_half_stop = df[(df['Behaviour'] == 'First half') & (df['B_Type'] == 'STOP')]['Time'].iloc[0]
        first_half_dur = (first_half_stop - first_half_start) / 60
        second_half_start = df[(df['Behaviour'] == 'Second half') & (df['B_Type'] == 'START')]['Time'].iloc[0]
        second_half_stop = df[(df['Behaviour'] == 'Second half') & (df['B_Type'] == 'STOP')]['Time'].iloc[0]
        second_half_dur = (second_half_stop - second_half_start) / 60
        
        # Get timeout intervals
        timeout_rows = df[df['Behaviour'] == 'Timeout']
        timeout_starts = timeout_rows[timeout_rows['B_Type'] == 'START']['Time'].tolist()
        timeout_stops  = timeout_rows[timeout_rows['B_Type'] == 'STOP']['Time'].tolist()
        timeout_intervals = list(zip(timeout_starts, timeout_stops))
        
        # Filter only Time_on_offense_teamA state rows
        a_offense_rows = df[df['Behaviour'] == 'Time_on_offense_teamA']

        # Get list of start/stop timestamps in order
        starts = a_offense_rows[a_offense_rows['B_Type'] == 'START']['Time'].tolist()
        stops  = a_offense_rows[a_offense_rows['B_Type'] == 'STOP']['Time'].tolist()

        # Safety check: they should match in length
        if len(starts) != len(stops):
            raise ValueError("Mismatched number of START and STOP events for Team A offense.")

        # Zip into intervals
        a_intervals = list(zip(starts, stops))

        # Filter only Time_on_offense_teamB state rows
        b_offense_rows = df[df['Behaviour'] == 'Time_on_offense_teamB']

        # Get list of start/stop timestamps in order
        starts = b_offense_rows[b_offense_rows['B_Type'] == 'START']['Time'].tolist()
        stops  = b_offense_rows[b_offense_rows['B_Type'] == 'STOP']['Time'].tolist()

        # Safety check: they should match in length
        if len(starts) != len(stops):
            raise ValueError("Mismatched number of START and STOP events for Team B offense.")

        # Zip into intervals
        b_intervals = list(zip(starts, stops))

        # Subtract timeouts from Team A offense
        a_intervals_clean = self.subtract_intervals(a_intervals, timeout_intervals)
        b_intervals_clean = self.subtract_intervals(b_intervals, timeout_intervals)

        # Sum offense duration team A
        a_total_minutes = sum(stop - start for start, stop in a_intervals_clean) / 60

        # Sum offense duration team B
        b_total_minutes = sum(stop - start for start, stop in b_intervals_clean) / 60
        
        total_timeout_duration = sum(stop - start for start, stop in timeout_intervals) / 60
        
        total_gametime_excl_breaks = first_half_dur + second_half_dur - total_timeout_duration/60
        
        df_teamA_offense = self.rows_in_intervals(df, a_intervals_clean)
        df_teamB_offense = self.rows_in_intervals(df, b_intervals_clean)
        
        df_teamA_offense = df_teamA_offense[df_teamA_offense['Behaviour'] != 'Time_on_offense_teamB']
        df_teamB_offense = df_teamB_offense[df_teamB_offense['Behaviour'] != 'Time_on_offense_teamA']

        # Save them for later use
        self.df_teamA_offense = df_teamA_offense
        self.df_teamB_offense = df_teamB_offense
        self.a_total_minutes = a_total_minutes
        self.b_total_minutes = b_total_minutes
        self.total_gametime_excl_breaks = total_gametime_excl_breaks
        self.a_intervals = a_intervals
        self.b_intervals = b_intervals
        self.timeout_intervals = timeout_intervals

        # return {"Total game time is:", total_gametime, 
        #         "minutes.\nTotal duration of the first half is:",
        #         first_half_dur, "minutes.\nTotal duration of the second half is:", 
        #         second_half_dur, "minutes.\nTeam A was on offense for:", a_total_minutes, 
        #         "minutes (excluding timeouts).\nTeam B was on offense for:", b_total_minutes, 
        #         "minutes (excluding timeouts).\nTotal timeout duration:", total_timeout_duration, 
        #         "minutes. \nTotal playing time of the whole game (excluding breaks) is:", total_gametime_excl_breaks, "minutes."
        #         }, df_teamA_offense, df_teamB_offense, a_total_minutes, b_total_minutes
        
        return df_teamA_offense, df_teamB_offense, a_total_minutes, b_total_minutes, total_gametime_excl_breaks, a_intervals, b_intervals, timeout_intervals

    def extract_indicators_teamA(self, df, game_id):
        # First call the compute function to populate instance variables
        self.compute_game_stats(df)
        # Now you can use them
        df_teamA_offense = self.df_teamA_offense
        a_total_minutes = self.a_total_minutes
        total_gametime_excl_breaks = self.total_gametime_excl_breaks
        a_intervals = self.a_intervals
        timeout_intervals = self.timeout_intervals
        
        indicators_A = {}

        key = f"{game_id}_teamA"
        
        #1
        scores_A = df_teamA_offense['Behaviour'].value_counts().get('Score', 0)
        att_scores_A = df_teamA_offense['Behaviour'].value_counts().get('Scoring_attempt', 0)
        f_turnover_EZ_A = df_teamA_offense[(df_teamA_offense['Behaviour'] == 'Forced_turnover') & (df_teamA_offense['Field_Loc'] == 'EZ_Off')].shape[0]
        uf_turnover_EZ_A = df_teamA_offense[(df_teamA_offense['Behaviour'] == 'Unforced_turnover') & (df_teamA_offense['Field_Loc'] == 'EZ_Off')].shape[0]
        score_attempts_A = scores_A + att_scores_A + f_turnover_EZ_A + uf_turnover_EZ_A
        
        #2
        score_efficiency_A = scores_A / score_attempts_A
        
        #3
        disc_possession_A = a_total_minutes / total_gametime_excl_breaks * 100

        #4
        succ_pass_A = df_teamA_offense['Behaviour'].value_counts().get('Successful_pass', 0) + scores_A
        f_turnover_A = df_teamA_offense['Behaviour'].value_counts().get('Forced_turnover', 0)
        uf_turnover_A = df_teamA_offense['Behaviour'].value_counts().get('Unforced_turnover', 0)
        total_pass_A = succ_pass_A + f_turnover_A + uf_turnover_A + att_scores_A
        pass_acc_A = succ_pass_A / total_pass_A * 100
        
        #7
        total_turnover_A = f_turnover_A + uf_turnover_A
        
        #10
        score_times = df_teamA_offense[df_teamA_offense['Behaviour'] == 'Score']['Time'].tolist()
        scoring_pos_A = self.get_scoring_possessions(a_intervals, score_times)
        passes_per_score_A = self.count_passes_in_possessions(scoring_pos_A, df_teamA_offense)
        avg_passes_per_score_A = np.mean(passes_per_score_A) if passes_per_score_A else 0
        
        #11
        pass_rate_A = total_pass_A / a_total_minutes
        
        #12
        teamA_durations = self.subtract_timeouts(a_intervals, timeout_intervals)
        avg_teamA_sec = np.mean(teamA_durations)
        
        indicators_A[key] = {
                "score_attempts": score_attempts_A,
                "score_efficiency": score_efficiency_A,
                "disc_possession": disc_possession_A,
                "pass_acc": pass_acc_A,
                "f_turnover": f_turnover_A,
                "uf_turnover": uf_turnover_A,
                "total_turnover": total_turnover_A,
                # "subs": df_teamA_offense['Subs_A'[0]],
                # "early_win": df_teamA_offense['Early win'[0]],
                "avg_passes_per_score": avg_passes_per_score_A,
                "pass_rate": pass_rate_A,
                "avg_poss": avg_teamA_sec,
        }

        return indicators_A
    
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

    def run(self):
        for filename in os.listdir(self.input_dir):
            if filename.endswith(".csv"):
                filepath = os.path.join(self.input_dir, filename)
                game_id = filename.split("_")[0]

                try:
                    df = pd.read_csv(filepath)
                    cleaned_df = self.clean_data(df)

                    # Save cleaned data
                    cleaned_path = os.path.join(self.output_dir, f"cleaned_{filename}")
                    cleaned_df.to_csv(cleaned_path, index=False)

                    # Extract indicators
                    indicators_A = self.extract_indicators_teamA(cleaned_df, game_id)
                    for team_key, team_indicators in indicators_A.items():
                        row = {"team_id": team_key}
                        row.update(team_indicators)
                        self.summary_data.append(row)

                    print(f"Processed {filename}")
                except Exception as e:
                    print(f"Error in {filename}: {e}")

        # Save summary indicators
        summary_df = pd.DataFrame(self.summary_data)
        summary_df.to_csv(os.path.join(self.output_dir, "summary_indicators.csv"), index=False)
        print("Saved summary_indicators.csv")

if __name__ == "__main__":
    input_dir = "./data/raw"
    output_dir = "./data/cleaned"

    cleaner = DataBatchCleaner(input_dir, output_dir)
    cleaner.run()