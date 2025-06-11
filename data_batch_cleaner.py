import numpy as np
import os
import pandas as pd

class DataBatchCleaner:
    def __init__(self, input_dir, output_dir):
        """Initializes the DataBatchCleaner instance with input and output directories where data files are read and saved."""
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.summary_data = []
        self.game_stats_data = []

    def clean_data(self, df):
        """Cleans and preprocesses the raw data by dropping useless columns and renaming the used columns for clarity."""
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
        """Subtracts a list of time intervals (sub_intervals) from a base set of intervals (base_intervals) and returns the remaining intervals."""
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
        """Filters and returns rows from a DataFrame that fall within the given list of time intervals."""
        return df[df['Time'].apply(lambda t: any(start <= t <= end for start, end in intervals))]
    
    def subtract_timeouts(self, possessions, timeouts):
        """Removes timeout intervals from the list of possession intervals and returns the adjusted time intervals."""
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
        """Returns a list of possession intervals during which scores occurred, based on score timestamps and known possession intervals."""
        scoring_possessions = []
        for start, stop in possessions:
            if any((score_time >= start) and (score_time <= stop) for score_time in score_events):
                scoring_possessions.append((start, stop))
        return scoring_possessions
    
    def count_passes_in_possessions(self, possessions, team_df):
        """Counts the number of passes made within each scoring possession, returning a list of pass counts."""
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
        """Computes and stores game statistics, such as duration of the games, times on offense for each team, etc."""
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
        
        game_stats = {
            "a_total_minutes": a_total_minutes,
            "b_total_minutes": b_total_minutes,
            "first_half_dur": first_half_dur,
            "second_half_dur": second_half_dur,
            "total_timeout_duration": total_timeout_duration,
            "total_gametime_excl_breaks": total_gametime_excl_breaks,
            "total_gametime": total_gametime
        }
        
        return df_teamA_offense, df_teamB_offense, a_total_minutes, b_total_minutes, total_gametime_excl_breaks, a_intervals, b_intervals, timeout_intervals, game_stats
    
    def extract_indicators_by_team(self, df, game_id, team):
        """Extracts performance indicators for a specified team in a given game."""
        self.compute_game_stats(df)

        if team == 'A':
            df_team = self.df_teamA_offense
            total_minutes = self.a_total_minutes
            intervals = self.a_intervals
        elif team == 'B':
            df_team = self.df_teamB_offense
            total_minutes = self.b_total_minutes
            intervals = self.b_intervals
        else:
            raise ValueError("Team must be 'A' or 'B'.")

        total_gametime_excl_breaks = self.total_gametime_excl_breaks
        timeout_intervals = self.timeout_intervals

        key = f"{game_id}_team{team}"

        # 1
        scores = df_team['Behaviour'].value_counts().get('Score', 0)
        att_scores = df_team['Behaviour'].value_counts().get('Scoring_attempt', 0)
        f_turnover_EZ = df_team[(df_team['Behaviour'] == 'Forced_turnover') & (df_team['Field_Loc'] == 'EZ_Off')].shape[0]
        uf_turnover_EZ = df_team[(df_team['Behaviour'] == 'Unforced_turnover') & (df_team['Field_Loc'] == 'EZ_Off')].shape[0]
        score_attempts = scores + att_scores + f_turnover_EZ + uf_turnover_EZ

        # 2
        score_efficiency = scores / score_attempts if score_attempts * 100 else 0

        # 3
        disc_possession = total_minutes / total_gametime_excl_breaks * 100

        # 4
        succ_pass = df_team['Behaviour'].value_counts().get('Successful_pass', 0) + scores
        f_turnover = df_team['Behaviour'].value_counts().get('Forced_turnover', 0)
        uf_turnover = df_team['Behaviour'].value_counts().get('Unforced_turnover', 0)
        total_pass = succ_pass + f_turnover + uf_turnover + att_scores
        pass_acc = succ_pass / total_pass * 100 if total_pass else 0

        # 7
        total_turnover = f_turnover + uf_turnover
        
        # 8
        subs = df_team[f'Subs_{team}'].iloc[0]
        
        # 9
        early_win = df_team['Early win'].iloc[0]

        # 10
        score_times = df_team[df_team['Behaviour'] == 'Score']['Time'].tolist()
        scoring_pos = self.get_scoring_possessions(intervals, score_times)
        passes_per_score = self.count_passes_in_possessions(scoring_pos, df_team)
        avg_passes_per_score = np.mean(passes_per_score) if passes_per_score else 0

        # 11
        pass_rate = total_pass / total_minutes if total_minutes else 0

        # 12
        durations = self.subtract_timeouts(intervals, timeout_intervals)
        avg_poss = np.mean(durations) if durations else 0
        
        # Determine win (1 = win, 0 = loss)
        a_scores = self.df_teamA_offense['Behaviour'].value_counts().get('Score', 0)
        b_scores = self.df_teamB_offense['Behaviour'].value_counts().get('Score', 0)
        win = 1 if (team == 'A' and a_scores > b_scores) or (team == 'B' and b_scores > a_scores) else 0

        return {
            key: {
                "win": win,
                "points_scored": scores,
                "score_attempts": score_attempts,
                "score_efficiency": score_efficiency,
                "disc_possession": disc_possession,
                "pass_acc": pass_acc,
                "f_turnover": f_turnover,
                "uf_turnover": uf_turnover,
                "total_turnover": total_turnover,
                "subs": subs,
                "early_win": early_win,
                "avg_passes_per_score": avg_passes_per_score,
                "pass_rate": pass_rate,
                "avg_poss": avg_poss,
            } 
        }

    def run(self):
        for filename in os.listdir(self.input_dir):
            if filename.endswith(".csv"):
                filepath = os.path.join(self.input_dir, filename)
                game_id = filename.split("_")[0]

                try:
                    df = pd.read_csv(filepath)
                    cleaned_df = self.clean_data(df)

                    # Save cleaned data of each game
                    cleaned_path = os.path.join(self.output_dir, f"cleaned_{filename}")
                    cleaned_df.to_csv(cleaned_path, index=False)
                    
                    # Extract stats
                    df_teamA_offense, df_teamB_offense, a_total_minutes, b_total_minutes, total_gametime_excl_breaks, a_intervals, b_intervals, timeout_intervals, game_stats = self.compute_game_stats(cleaned_df)

                    # 💡 Add game_id to game_stats and store
                    game_stats["game_id"] = game_id
                    self.game_stats_data.append(game_stats)

                    # Extract indicators
                    for team in ['A', 'B']:
                        indicators = self.extract_indicators_by_team(cleaned_df, game_id, team)
                        for team_key, team_indicators in indicators.items():
                            row = {"team_id": team_key}
                            row.update(team_indicators)
                            self.summary_data.append(row)

                    print(f"Processed {filename}")
                except Exception as e:
                    print(f"Error in {filename}: {e}")

        # Save summary indicators to a CSV file
        summary_df = pd.DataFrame(self.summary_data)
        summary_df = summary_df.sort_values(by = 'team_id').reset_index(drop = True)
        summary_df.to_csv(os.path.join(self.output_dir, "summary_indicators.csv"), index=False)
        
        # Save game-level statistics
        game_stats_df = pd.DataFrame(self.game_stats_data)
        cols = ['game_id'] + [col for col in game_stats_df.columns if col != 'game_id'] # 💡 Move 'game_id' column to the front
        game_stats_df = game_stats_df[cols]
        game_stats_df = game_stats_df.sort_values(by='game_id').reset_index(drop=True)
        game_stats_df.to_csv(os.path.join(self.output_dir, "game_stats.csv"), index=False)

        print("Saved summary_indicators.csv and game_stats.csv")

if __name__ == "__main__":
    input_dir = "./data/raw"
    output_dir = "./data/cleaned"

    cleaner = DataBatchCleaner(input_dir, output_dir)
    cleaner.run()