import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

class ProbMatrixBuilder:
    def __init__(self, input_dir, output_dir):
        """Initializes the DataBatchCleaner instance with input and output directories where data files are read and saved."""
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.all_winners = []
        self.all_losers = []
        
    def switch_off_def(self, value):
        """Define a function to switch _Off and _Def"""
        if isinstance(value, str):
            if value.endswith('_Off'):
                return value.replace('_Off', '_Def')
            elif value.endswith('_Def'):
                return value.replace('_Def', '_Off')
        return value
    
    def rows_in_intervals(self, df, intervals):
        """Filters and returns rows from a DataFrame that fall within the given list of time intervals."""
        return df[df['Time'].apply(lambda t: any(start <= t <= end for start, end in intervals))]
    
    def row_normalize(self, matrix):
        return matrix.div(matrix.sum(axis=1), axis=0).fillna(0)
        
    def prepare_data(self, df):
        """Prepares data by dropping useless columns and renaming the used columns for clarity."""
        df = df.iloc[:, [0,1,2,3,4,6,7,8,5]]
        df = df.drop(columns=['Observation id', 'Subs_A', 'Subs_B', 'Early win', 'Comment'])
        df = df.rename(columns={"Field_Loc": "Throw_To"})
        df = df[~df['Behaviour'].isin(['First half', 'Second half', 'Timeout'])]
        df.loc[df['Behaviour'] == 'Score', 'Throw_To'] = 'EZ_Off'
        
        # Save the rows to hide and their original index so we can restore them later
        hidden_rows = df[df['Behaviour'].isin(['Time_on_offense_teamA', 'Time_on_offense_teamB'])]
        hidden_rows = hidden_rows.copy()
        hidden_rows['original_index'] = hidden_rows.index

        # Filter out those rows from the main DataFrame, add a column and fill values
        visible_df = df[~df['Behaviour'].isin(['Time_on_offense_teamA', 'Time_on_offense_teamB'])].copy()
        visible_df.loc[:, 'Throw_From'] = visible_df['Throw_To'].shift(1)
        
        # Define the turnover conditions
        turnover_mask = visible_df['Behaviour'].isin(['Forced_turnover', 'Unforced_turnover'])

        # Shift the turnover mask down by 1 to affect the row after the turnover
        rows_to_modify = turnover_mask.shift(fill_value=False)

        # Apply the switch only to rows where the row above was a turnover
        visible_df.loc[rows_to_modify, 'Throw_From'] = visible_df.loc[rows_to_modify, 'Throw_From'].apply(self.switch_off_def)

        # Add the same column to the hidden rows df
        hidden_rows['Throw_From'] = None  # or np.nan, or any placeholder value

        # Combine the dataframes
        restored_df = pd.concat([visible_df, hidden_rows.drop(columns='original_index')])

        # Sort by the original index to restore the original row order
        df = restored_df.sort_index()
        
        return df
    
    def prepare_teams(self, df):
        # Filter only Time_on_offense_teamA state rows
        a_offense_rows = df[df['Behaviour'] == 'Time_on_offense_teamA']

        # Get list of start/stop timestamps in order
        starts = a_offense_rows[a_offense_rows['B_Type'] == 'START']['Time'].tolist()
        stops  = a_offense_rows[a_offense_rows['B_Type'] == 'STOP']['Time'].tolist()

        # Zip into intervals
        a_intervals = list(zip(starts, stops))

        # Filter only Time_on_offense_teamB state rows
        b_offense_rows = df[df['Behaviour'] == 'Time_on_offense_teamB']

        # Get list of start/stop timestamps in order
        starts = b_offense_rows[b_offense_rows['B_Type'] == 'START']['Time'].tolist()
        stops  = b_offense_rows[b_offense_rows['B_Type'] == 'STOP']['Time'].tolist()

        # Zip into intervals
        b_intervals = list(zip(starts, stops))

        df_teamA = self.rows_in_intervals(df, a_intervals)
        df_teamB = self.rows_in_intervals(df, b_intervals)

        df_teamA = df_teamA[~df_teamA['Behaviour'].isin(['Time_on_offense_teamA', 'Time_on_offense_teamB'])]
        df_teamB = df_teamB[~df_teamB['Behaviour'].isin(['Time_on_offense_teamA', 'Time_on_offense_teamB'])]
        
        # Determine win (1 = win, 0 = loss)
        a_scores = df_teamA['Behaviour'].value_counts().get('Score', 0)
        b_scores = df_teamB['Behaviour'].value_counts().get('Score', 0)
        
        winner_team = 'A' if a_scores > b_scores else 'B'
        
        return df_teamA, df_teamB, winner_team
    
    def probability_matrix(self, df_team):
        counts = df_team.groupby(['Throw_From', 'Throw_To']).size().unstack(fill_value=0)
        return counts
        
    def plot_matrix(self, matrix, title, filename, cmap, center=None):
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, cmap=cmap, annot=True, fmt='g', center=center)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def run(self):
        for filename in os.listdir(self.input_dir):
            if filename.startswith("cleaned_"):
                filepath = os.path.join(self.input_dir, filename)
                game_id = os.path.splitext(filename)[0]

                try:
                    df = pd.read_csv(filepath)
                    df = self.prepare_data(df)
                    df_teamA, df_teamB, winner = self.prepare_teams(df)

                    matrixA = self.probability_matrix(df_teamA)
                    matrixB = self.probability_matrix(df_teamB)

                    if winner == 'A':
                        self.all_winners.append(matrixA)
                        self.all_losers.append(matrixB)
                        self.plot_matrix(matrixA, f"{game_id} - Winners (Team A)", f"{game_id}_winners.png", 'Blues')
                        self.plot_matrix(matrixB, f"{game_id} - Losers (Team B)", f"{game_id}_losers.png", 'Reds')
                    else:
                        self.all_winners.append(matrixB)
                        self.all_losers.append(matrixA)
                        self.plot_matrix(matrixB, f"{game_id} - Winners (Team B)", f"{game_id}_winners.png", 'Blues')
                        self.plot_matrix(matrixA, f"{game_id} - Losers (Team A)", f"{game_id}_losers.png", 'Reds')

                    print(f"Processed {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

        # Combine all winners and losers matrices (absolute counts)
        total_winners = sum(self.all_winners).fillna(0)
        total_losers = sum(self.all_losers).fillna(0)

        # Save absolute count heatmaps
        self.plot_matrix(total_winners, "Combined Winners (Counts)", "all_winners_counts.png", 'Blues')
        self.plot_matrix(total_losers, "Combined Losers (Counts)", "all_losers_counts.png", 'Reds')

        # Normalize for probability (percentage) matrices
        winners_prob = self.row_normalize(total_winners)
        losers_prob = self.row_normalize(total_losers)

        self.plot_matrix(winners_prob, "Combined Winners (Percentages)", "all_winners_percentages.png", 'Blues')
        self.plot_matrix(losers_prob, "Combined Losers (Percentages)", "all_losers_percentages.png", 'Reds')

        # Difference matrix (winners - losers)
        diff_matrix = total_winners.subtract(total_losers, fill_value=0)
        self.plot_matrix(diff_matrix, "Difference Matrix (Winners - Losers) (absolute values)", "difference_matrix.png", 'RdBu', center=0)

        print("All matrices saved.")

if __name__ == "__main__":
    input_dir = "./data/cleaned"
    output_dir = "./data/probability_analysis"

    builder = ProbMatrixBuilder(input_dir, output_dir)
    builder.run()