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
    
    @staticmethod
    def subtract_intervals(base_intervals, sub_intervals):
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
    
    def rows_in_intervals(df, intervals):
        return df[df['Time'].apply(lambda t: any(start <= t <= end for start, end in intervals))]
    
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
        a_total_seconds = sum(stop - start for start, stop in a_intervals_clean)
        a_total_minutes = a_total_seconds/60

        # Sum offense duration team B
        b_total_seconds = sum(stop - start for start, stop in b_intervals_clean)
        b_total_minutes = b_total_seconds/60
        
        total_timeout_duration = sum(stop - start for start, stop in timeout_intervals) / 60
        
        total_gametime_excl_breaks = first_half_dur + second_half_dur - total_timeout_duration/60
        
        df_teamA_offense = self.rows_in_intervals(df, a_intervals_clean)
        df_teamB_offense = self.rows_in_intervals(df, b_intervals_clean)
        
        df_teamA_offense = df_teamA_offense[df_teamA_offense['Behaviour'] != 'Time_on_offense_teamB']
        df_teamB_offense = df_teamB_offense[df_teamB_offense['Behaviour'] != 'Time_on_offense_teamA']

        return {"Total game time is:", total_gametime, 
                "minutes.\nTotal duration of the first half is:",
                first_half_dur, "minutes.\nTotal duration of the second half is:", 
                second_half_dur, "minutes.\nTeam A was on offense for:", a_total_minutes, 
                "minutes (excluding timeouts).\nTeam B was on offense for:", b_total_minutes, 
                "minutes (excluding timeouts).\nTotal timeout duration:", total_timeout_duration, 
                "minutes. \nTotal playing time of the whole game (excluding breaks) is:", total_gametime_excl_breaks, "minutes."
                }

    def extract_indicators(self, df, match_id):
        teams = df['Team'].unique()  # or replace with the actual column name
        indicators = {}

        for team in teams:
            team_df = df[df['Team'] == team]
            key = f"{match_id}_{team}"

            total_game_time = (df['Time'].max() - df['Time'].min()) / 60
            scores = team_df[team_df['Behaviour'] == 'Score'].shape[0]
            passes = team_df[team_df['Behaviour'] == 'Pass'].shape[0]
            shots = team_df[team_df['Behaviour'] == 'Shot'].shape[0]
            steals = team_df[team_df['Behaviour'] == 'Steal'].shape[0]
            # You can add more indicators here...

            indicators[key] = {
                "total_game_time": total_game_time,
                "score_count": scores,
                "pass_count": passes,
                "shot_count": shots,
                "steal_count": steals,
                # Add more indicators...
            }

        return indicators

    def run(self):
        for filename in os.listdir(self.input_dir):
            if filename.endswith(".csv"):
                filepath = os.path.join(self.input_dir, filename)
                match_id = filename.split("_")[0]

                try:
                    df = pd.read_csv(filepath)
                    cleaned_df = self.clean_data(df)

                    # Save cleaned data
                    cleaned_path = os.path.join(self.output_dir, f"cleaned_{filename}")
                    cleaned_df.to_csv(cleaned_path, index=False)

                    # Extract indicators
                    indicators = self.extract_indicators(cleaned_df, match_id)
                    for team_key, team_indicators in indicators.items():
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
