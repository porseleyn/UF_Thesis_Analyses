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
            turnover_type = None
            if df.iloc[i]['Behaviour'] == 'Pull_pickup':
                from_zone = 'Pull'
                to_zone = df.iloc[i]['Throw_To']

            elif df.iloc[i]['Behaviour'] in ['Successful_pass', 'Unsuccessful_pass']:
                if i > 0 and df.iloc[i - 1]['Behaviour'] in ['Forced_turnover', 'Unforced_turnover']:
                    from_zone = 'Earned Turnover'
                else:
                    from_zone = df.iloc[i]['Throw_From']
                to_zone = df.iloc[i]['Throw_To']

            elif df.iloc[i]['Behaviour'] == 'Score':
                from_zone = df.iloc[i]['Throw_From']
                to_zone = 'Score'

            elif df.iloc[i]['Behaviour'] in ['Forced_turnover', 'Unforced_turnover']:
                from_zone = df.iloc[i]['Throw_From']
                to_zone = 'Turnover'
                turnover_type = 'forced' if df.iloc[i]['Behaviour'] == 'Forced_turnover' else 'unforced'

            else:
                continue

            transitions.append((from_zone, to_zone, turnover_type))
        return transitions

    def build_matrix(self, transitions):
        matrix = pd.DataFrame(0, index=self.row_labels, columns=self.col_labels)
        forced = pd.DataFrame(0, index=self.row_labels, columns=self.col_labels)
        unforced = pd.DataFrame(0, index=self.row_labels, columns=self.col_labels)

        for from_zone, to_zone, turnover_type in transitions:
            if from_zone in matrix.index and to_zone in matrix.columns:
                matrix.at[from_zone, to_zone] += 1
                if to_zone == 'Turnover':
                    if turnover_type == 'forced':
                        forced.at[from_zone, to_zone] += 1
                    elif turnover_type == 'unforced':
                        unforced.at[from_zone, to_zone] += 1

        normalized = self.row_normalize(matrix)

        annotated = pd.DataFrame("", index=self.row_labels, columns=self.col_labels)
        for row in self.row_labels:
            row_sum = matrix.loc[row].sum()
            for col in self.col_labels:
                total = normalized.at[row, col]
                if row_sum > 0 and col == 'Turnover':
                    f = forced.at[row, col] / row_sum
                    uf = unforced.at[row, col] / row_sum
                    annotated.at[row, col] = f"{total:.6f}\n(F{f:.3f} | U{uf:.3f})"
                elif total >= 0 or col == 'Turnover':
                    annotated.at[row, col] = f"{total:.6f}"
        return normalized, annotated

    def build_raw_matrix(self, transitions):
        matrix = pd.DataFrame(0, index=self.row_labels, columns=self.col_labels)
        for from_zone, to_zone, _ in transitions:
            if from_zone in matrix.index and to_zone in matrix.columns:
                matrix.at[from_zone, to_zone] += 1
        return matrix

    def plot_heatmap(self, matrix, title, filename, cmap="viridis", center=None, forbidden=[], annot=None, fmt=".0f"):
        plt.figure(figsize=(12, 10))
        mask = pd.DataFrame(False, index=matrix.index, columns=matrix.columns)

        for from_zone, to_zone in forbidden:
            if from_zone in mask.index and to_zone in mask.columns:
                mask.loc[from_zone, to_zone] = True

        ax = sns.heatmap(
            matrix,
            annot=annot if annot is not None else True,
            fmt="s" if annot is not None else fmt,
            cmap=cmap,
            center=center,
            vmin=-matrix.abs().max().max() if center is not None else None,
            vmax=matrix.abs().max().max() if center is not None else None,
            mask=mask,
            cbar=True,
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

        forbidden_transitions = [('Pull', 'Turnover'), ('Pull', 'Score')]

        # Raw and difference
        winner_raw = self.build_raw_matrix(self.all_winners)
        loser_raw = self.build_raw_matrix(self.all_losers)
        diff_raw = winner_raw.subtract(loser_raw, fill_value=0)

        # Normalized with annotation
        winner_matrix, winner_annot = self.build_matrix(self.all_winners)
        loser_matrix, loser_annot = self.build_matrix(self.all_losers)

        self.plot_heatmap(winner_matrix, "Winning Teams Transition Probabilities", "winners_heatmap.png",
                        cmap="Blues", forbidden=forbidden_transitions, annot=winner_annot)

        self.plot_heatmap(loser_matrix, "Losing Teams Transition Probabilities", "losers_heatmap.png",
                        cmap="Reds", forbidden=forbidden_transitions, annot=loser_annot)

        self.plot_heatmap(diff_raw, "Raw Count Difference Matrix (Winning Teams - Losing Teams)",
                        "difference_heatmap_raw_counts.png", cmap="RdBu", forbidden=forbidden_transitions, center=0)

        print("All heatmaps saved.")

if __name__ == "__main__":
    input_dir = "./data/cleaned"
    output_dir = "./data/probability_analysis/lam"

    builder = ProbMatrixBuilder(input_dir, output_dir)
    builder.run()
