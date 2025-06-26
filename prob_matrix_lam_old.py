import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

class ProbMatrixBuilder:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.all_winners = []
        self.all_losers = []

        self.zones = ['EZ_Def', 'RZ_Def', 'Mid_Def', 'Mid_Off', 'RZ_Off', 'EZ_Off']

        self.row_labels = ['Pull', 'Earned Turnover'] + [z for z in self.zones if z != 'EZ_Off']
        self.col_labels = [z for z in self.zones if z != 'EZ_Off'] + ['Turnover', 'Score']

    def switch_off_def(self, value):
        if isinstance(value, str):
            if value.endswith('_Off'):
                return value.replace('_Off', '_Def')
            elif value.endswith('_Def'):
                return value.replace('_Def', '_Off')
        return value

    def rows_in_intervals(self, df, intervals):
        return df[df['Time'].apply(lambda t: any(start <= t <= end for start, end in intervals))]

    def row_normalize(self, matrix):
        return matrix.div(matrix.sum(axis=1), axis=0).fillna(0)

    def prepare_data(self, df):
        df = df.iloc[:, [0,1,2,3,4,6,7,8,5]]
        df = df.drop(columns=['Observation id', 'Subs_A', 'Subs_B', 'Early win', 'Comment'])
        df = df.rename(columns={"Field_Loc": "Throw_To"})
        df = df[~df['Behaviour'].isin(['First half', 'Second half', 'Timeout'])]
        df.loc[df['Behaviour'] == 'Score', 'Throw_To'] = 'EZ_Off'

        hidden_rows = df[df['Behaviour'].isin(['Time_on_offense_teamA', 'Time_on_offense_teamB'])]
        hidden_rows = hidden_rows.copy()
        hidden_rows['original_index'] = hidden_rows.index

        visible_df = df[~df['Behaviour'].isin(['Time_on_offense_teamA', 'Time_on_offense_teamB'])].copy()
        visible_df.loc[:, 'Throw_From'] = visible_df['Throw_To'].shift(1)

        turnover_mask = visible_df['Behaviour'].isin(['Forced_turnover', 'Unforced_turnover'])
        rows_to_modify = turnover_mask.shift(fill_value=False)
        visible_df.loc[rows_to_modify, 'Throw_From'] = visible_df.loc[rows_to_modify, 'Throw_From'].apply(self.switch_off_def)

        hidden_rows['Throw_From'] = None
        restored_df = pd.concat([visible_df, hidden_rows.drop(columns='original_index')])
        df = restored_df.sort_index()

        return df

    def prepare_teams(self, df):
        a_rows = df[df['Behaviour'] == 'Time_on_offense_teamA']
        a_intervals = list(zip(a_rows[a_rows['B_Type'] == 'START']['Time'], a_rows[a_rows['B_Type'] == 'STOP']['Time']))
        b_rows = df[df['Behaviour'] == 'Time_on_offense_teamB']
        b_intervals = list(zip(b_rows[b_rows['B_Type'] == 'START']['Time'], b_rows[b_rows['B_Type'] == 'STOP']['Time']))

        df_teamA = self.rows_in_intervals(df, a_intervals)
        df_teamB = self.rows_in_intervals(df, b_intervals)

        df_teamA = df_teamA[~df_teamA['Behaviour'].isin(['Time_on_offense_teamA', 'Time_on_offense_teamB'])]
        df_teamB = df_teamB[~df_teamB['Behaviour'].isin(['Time_on_offense_teamA', 'Time_on_offense_teamB'])]

        a_scores = df_teamA['Behaviour'].value_counts().get('Score', 0)
        b_scores = df_teamB['Behaviour'].value_counts().get('Score', 0)

        winner_team = 'A' if a_scores > b_scores else 'B'
        return df_teamA, df_teamB, winner_team

    def extract_transitions(self, df):
        transitions = []
        df = df.reset_index(drop=True)

        for i in range(len(df)):
            if df.iloc[i]['Behaviour'] == 'Pull_pickup':
                from_zone = 'Pull'
                to_zone = df.iloc[i]['Throw_To']
                transitions.append((from_zone, to_zone))

            elif df.iloc[i]['Behaviour'] in ['Successful_pass', 'Unsuccessful_pass']:
                if i > 0 and df.iloc[i - 1]['Behaviour'] in ['Forced_turnover', 'Unforced_turnover']:
                    from_zone = 'Earned Turnover'
                else:
                    from_zone = df.iloc[i]['Throw_From']
                to_zone = df.iloc[i]['Throw_To']
                transitions.append((from_zone, to_zone))

            elif df.iloc[i]['Behaviour'] == 'Score':
                from_zone = df.iloc[i]['Throw_From']
                transitions.append((from_zone, 'Score'))

            elif df.iloc[i]['Behaviour'] in ['Forced_turnover', 'Unforced_turnover']:
                from_zone = df.iloc[i]['Throw_From']
                transitions.append((from_zone, 'Turnover'))

        return transitions

    def build_matrix(self, transitions):
        matrix = pd.DataFrame(0, index=self.row_labels, columns=self.col_labels)
        for from_zone, to_zone in transitions:
            if from_zone in matrix.index and to_zone in matrix.columns:
                matrix.at[from_zone, to_zone] += 1
        return self.row_normalize(matrix)
    
    def build_raw_matrix(self, transitions):
        matrix = pd.DataFrame(0, index=self.row_labels, columns=self.col_labels)
        for from_zone, to_zone in transitions:
            if from_zone in matrix.index and to_zone in matrix.columns:
                matrix.at[from_zone, to_zone] += 1
        return matrix


    def plot_heatmap(self, matrix, title, filename, cmap="viridis", center=None, forbidden=[]):
        plt.figure(figsize=(12, 10))

        # Create a mask of the same shape as the matrix, defaulting to False
        mask = pd.DataFrame(False, index=matrix.index, columns=matrix.columns)

        # Set True for all forbidden (from_zone, to_zone) transitions
        for from_zone, to_zone in forbidden:
            if from_zone in mask.index and to_zone in mask.columns:
                mask.loc[from_zone, to_zone] = True

        # Automatically set vmin and vmax for diverging colormaps
        if cmap in ["RdBu", "seismic", "coolwarm", "bwr"] and center is not None:
            max_abs = matrix.abs().max().max()
            sns.heatmap(
                matrix,
                annot=True,
                fmt='g',
                cmap=cmap,
                center=center,
                vmin=-max_abs,
                vmax=max_abs,
                mask=mask
            )
        else:
            sns.heatmap(
                matrix,
                annot=True,
                fmt='g',
                cmap=cmap,
                mask=mask
            )

        plt.title(title)
        plt.xlabel("Throw To", fontsize=12)
        plt.ylabel("Throw From", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()


    def run(self):
        for file in os.listdir(self.input_dir):
            if not file.startswith("cleaned_"):
                continue

            df = pd.read_csv(os.path.join(self.input_dir, file))
            df = self.prepare_data(df)
            df_teamA, df_teamB, winner = self.prepare_teams(df)

            transitions_A = self.extract_transitions(df_teamA)
            transitions_B = self.extract_transitions(df_teamB)

            if winner == 'A':
                self.all_winners += transitions_A
                self.all_losers += transitions_B
            else:
                self.all_winners += transitions_B
                self.all_losers += transitions_A

        # Raw count matrices
        winner_raw = self.build_raw_matrix(self.all_winners)
        loser_raw = self.build_raw_matrix(self.all_losers)
        diff_raw = winner_raw.subtract(loser_raw, fill_value=0)

        # Build matrices
        winner_matrix = self.build_matrix(self.all_winners)
        loser_matrix = self.build_matrix(self.all_losers)

        # Save heatmaps
        forbidden_transitions = [('Pull', 'Turnover'), ('Pull', 'Score')]
        self.plot_heatmap(winner_matrix, "Winner Transition Probabilities", "winners_heatmap.png", cmap="Blues", forbidden=forbidden_transitions)
        self.plot_heatmap(loser_matrix, "Loser Transition Probabilities", "losers_heatmap.png", cmap="Reds", forbidden=forbidden_transitions)
        self.plot_heatmap(diff_raw, "Raw Count Difference Matrix (Winners - Losers)", "difference_heatmap_raw_counts.png", cmap="RdBu", center=0)

        # # Save to CSV
        # winner_matrix.to_csv(os.path.join(self.output_dir, 'winner_matrix.csv'))
        # loser_matrix.to_csv(os.path.join(self.output_dir, 'loser_matrix.csv'))
        # diff_raw.to_csv(os.path.join(self.output_dir, 'difference_matrix_raw_counts.csv'))
        
        print("All heatmaps saved.")

if __name__ == "__main__":
    input_dir = "./data/cleaned"
    output_dir = "./data/probability_analysis/lam_old"

    builder = ProbMatrixBuilder(input_dir, output_dir)
    builder.run()