import math
import textwrap
import pandas as pd
import statsmodels.api
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import levene, shapiro, ranksums, rankdata, spearmanr
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor
os.environ['R_HOME'] = "C:\Program Files\R\R-4.4.2"
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr, data
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
import config as cfg
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings(
    "ignore",
    message="Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect."
)

groups = {
    # "P101": "1",
    # "P102": "2",
    # "P103": "3",
    "P104_01": "4",
    "P104_02": "5",
    "P104_03": "6",
    "P104_04": "7",
    "GQ01_01": "8\_1",
    "GQ01_02": "9\_1",
    "GQ01_03": "10\_1",
    "GQ01_04": "11\_1",
    "GQ02_01": "8\_2",
    "GQ02_02": "9\_2",
    "GQ02_03": "10\_2",
    "GQ02_04": "11\_2",
    "GQ03_01": "8\_3",
    "GQ03_02": "9\_3",
    "GQ03_03": "10\_3",
    "GQ03_04": "11\_3",
    "EQ01_01": "12\_1",
    "EQ01_02": "13\_1",
    "EQ01_03": "14\_1",
    # "EQ01_04": "15\_1",
    "EQ01_05": "16\_1",
    "EQ02_01": "12\_2",
    "EQ02_02": "13\_2",
    "EQ02_03": "14\_2",
    # "EQ02_04": "15\_2",
    "EQ02_05": "16\_2",
    "EQ03_01": "12\_3",
    "EQ03_02": "13\_3",
    "EQ03_03": "14\_3",
    # "EQ03_04": "15\_3",
    "EQ03_05": "16\_3",
    "BQ01_01": "17\_1",
    "BQ01_02": "18\_1",
    "BQ01_03": "19\_1",
    "BQ01_04": "20\_1",
    # "BQ01_05": "21\_1",
    "BQ02_01": "17\_2",
    "BQ02_02": "18\_2",
    "BQ02_03": "19\_2",
    "BQ02_04": "20\_2",
    # "BQ02_05": "21\_2",
    "BQ03_01": "17\_3",
    "BQ03_02": "18\_3",
    "BQ03_03": "19\_3",
    "BQ03_04": "20\_3",
    # "BQ03_05": "21\_3",
    "F101_01": "22\_3",
    "F101_02": "23",
    "F101_03": "24",
    "F101_04": "25",
    "F101_05": "26",
    "GQ01_mean": "8 mean",
    "GQ02_mean": "9 mean",
    "GQ03_mean": "10 mean",
    "GQ04_mean": "11 mean",
    "EQ01_mean": "12 mean",
    "EQ02_mean": "13 mean",
    "EQ03_mean": "14 mean",
    "EQ04_mean": "15 mean",
    "EQ05_mean": "16 mean",
    "BQ01_mean": "17 mean",
    "BQ02_mean": "18 mean",
    "BQ03_mean": "19 mean",
    "BQ04_mean": "20 mean",
    "BQ05_mean": "21 mean"
}
groups_reworked = {
    # "P101": "1",
    # "P102": "2",
    # "P103": "3",
    # "P104_01": "4",
    "P104_02": "experience in healthcare",
    "P104_03": "familiarity with ML",
    "P104_04": "awareness of XAI",
    "P104_combined": "familiarity with ML and XAI",
    "GQ01_mean": "G\\_U\\_1",
    "GQ02_mean": "G\\_U\\_2",
    "GQ03_mean": "G\\_U\\_3",
    "GQ04_mean": "G\\_H\\_1",
    "EQ01_mean": "E\\_U\\_1",
    "EQ02_mean": "E\\_U\\_2",
    "EQ03_mean": "E\\_H\\_1",
    #"EQ04_mean": "E\\_A\\_1",
    "EQ05_mean": "E\\_H\\_2",
    "BQ01_mean": "B\\_S\\_1",
    "BQ02_mean": "B\\_H\\_1",
    "BQ03_mean": "B\\_U\\_1",
    "BQ04_mean": "B\\_T\\_1",
    #"BQ05_mean": "B\\_A\\_1",
    "F101_01": "F\\_U\\_1",
    "F101_02": "F\\_U\\_2",
    "F101_03": "F\\_U\\_3",
    "F101_04": "F\\_H\\_1",
    "F101_05": "F\\_T\\_1",
}

def add_average(df, column_names, name):
    """
    mean of columns specified in a new column added to existing df
    :param df:
    :param column_names:
    :param name:
    :return:
    """
    df[name] = df[column_names].mean(axis=1)
    return df


def round_and_format(value):
    if pd.isna(value):
        return value
    elif value < 0.01:
        return "<0.01"
    else:
        return round(value, 2)


reversed_groups_reworked = {v: k for k, v in groups_reworked.items()}


def wilcoxon_ranksum(df, column_name):
    """
    Wilcoxon ranksum test per statement.
    :param df: study data
    :param column_name: statement name
    :return: p-value of the test
    """
    ### check assumptions for t-test
    # scores are independent, as they are from different people
    # data is measured at interval level: likert scales
    # normality
    test_assumptions = False
    if test_assumptions:
        # same length needed; groups slightly imbalanced; omitting 2 values won't hurt the assumption test
        group1 = df[column_name][df['G101'] == 1].values[:31]
        group2 = df[column_name][df['G101'] == 2].values[:31]
        differences = group1 - group2
        stat, shap_p_value = shapiro(differences, nan_policy='omit')
        text = f"Assumptions for {groups_reworked[column_name]}: \\\\ Shapiro-Wilk Test: $p$ = {round(shap_p_value,2)}, "
        if shap_p_value > 0.05:
            text += f"{round(shap_p_value,2)} > 0.05, thus the assumption of normally distributed data holds.\\\\"
        else:
            text += f"{round(shap_p_value,2)} < 0.05, thus the assumption of normally distributed data does not hold.\\\\"
        print(column_name, ' shapiro: ', shap_p_value)
        if shap_p_value < 0.05:
            print('shapiro failed')
        else:
            print('shapiro ok')
        # homogeneity of variance:
        stat, p_value = levene(group1, group2)
        print(f"Levene: W = {stat:.3f}, p-value = {p_value:.3f}")
        text += f"Levene's Test: $p$ = {round(p_value,2)}\\\\"
        if p_value > 0.05:
            print("levene ok")
            text += f"{round(p_value,2)} > 0.05, thus homogeneity of variance holds.\\\\"
        else:
            text += f"{round(p_value,2)} < 0.05, thus homogeneity of variance does not hold.\\\\"
            print("levene failed.")
        text += "\\\\"
        file_path = 'wilcoxon_assumptions.text'
        if os.path.exists(file_path):
            with open(file_path, 'a', encoding='utf-8') as file:
                file.write(text)
        else:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(text)
        return 0.5
    else:
        group1 = df[df['G101'] == 1][column_name]
        group2 = df[df['G101'] == 2][column_name]
        stat, p = ranksums(group1, group2, nan_policy='omit')
        full = np.concatenate([group1, group2])
        ranks = rankdata(full, nan_policy='omit')
        group1_ranks = ranks[:len(group1)]
        group2_ranks = ranks[len(group1):]
        W1 = np.sum(group1_ranks)
        W2 = np.sum(group2_ranks)
        if W1 < W2:
            W = W1
        else:
            W = W2
        N = len(group1) + len(group2)
        r = stat / np.sqrt(N)
        print(column_name, ' Wilcoxon: ', p)
        medians = df.groupby('G101')[column_name].median()
        med_1 = medians[1]
        med_2 = medians[2]

        statement_id = groups_reworked[column_name]

        if p < 0.05:
            result = f"The rating of statement {statement_id} ($Mdn={round(med_1, 2)}$) in the group `This' differed significantly the group `DDxT' ($Mdn={round(med_2, 2)}$), $W={round(W, 2)}$, $p={round(p, 2)}$, $r={round(r, 2)}$."
        else:
            result = f"The rating of statement {statement_id} ($Mdn={round(med_1, 2)}$) in the group `This' did not differ significantly the group `DDxT' ($Mdn={round(med_2, 2)}$), $W={round(W, 2)}$, $p={round(p, 2)}$, $r={round(r, 2)}$."
        file_path = os.path.join(cfg.TEXTS, column_name + '.txt')
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(result)
        plt.figure(figsize=(6, 6))
        sns.boxplot(x='G101', y=column_name, data=df, palette='pastel', hue=None, linewidth=2.5)
        sns.stripplot(x='G101', y=column_name, data=df, color='black', alpha=0.6, jitter=True, size=6)
        plt.xticks(ticks=[0, 1], labels=['DDxT', 'This'], fontsize=24)
        plt.yticks(fontsize=24)
        plt.xlabel('Group', labelpad=10, fontsize=26)
        plt.ylabel('5-point Likert Scale', labelpad=10, fontsize=26)
        plt.subplots_adjust(left=0.2, bottom=0.16)
        plt.savefig(os.path.join(cfg.IMGS, column_name + '.png'))
        # plt.show()
        plt.close('all')
        return p


def add_asterisks(row):
    """
    adds * for the level of significance in MLR tables
    :param row:
    :return:
    """
    p_value = row['P>|z|']
    if p_value < 0.001:
        return f"{row['B (SE)']} ***"
    elif p_value < 0.01:
        return f"{row['B (SE)']} **"
    elif p_value < 0.05:
        return f"{row['B (SE)']} *"
    else:
        return row['B (SE)']


def multinomial_log_reg(df, column_name, test_assumptions):
    """
    Multinomial logistic regression per statement name and study results.
    :param df: study data
    :param column_name: statement name
    :param test_assumptions: whether the assumptions for this test should be checked
    :return:
    """
    print(column_name)
    data = df
    if test_assumptions:
        # linearity of the logit (Discovering Statistics using R p.344,345
        data['log_P104_02'] = np.log(df['P104_02'].replace(0, np.nan))
        data['log_P104_03'] = np.log(df['P104_03'].replace(0, np.nan))
        data['log_P104_04'] = np.log(df['P104_04'].replace(0, np.nan))
        data['P104_02_logInt'] = data['P104_02'] * data['log_P104_02']
        data['P104_03_logInt'] = data['P104_03'] * data['log_P104_03']
        data['P104_04_logInt'] = data['P104_04'] * data['log_P104_04']
        data[column_name] = data[column_name].apply(lambda x: 1 if x > 2.5 else 0)
        data[column_name].astype('category')
        formula_with_interactions = f"{column_name} ~ P104_02 + P104_03 + P104_04 + P104_02_logInt + P104_03_logInt + P104_04_logInt"
        model_with_interactions = statsmodels.formula.api.mnlogit(formula_with_interactions, data).fit(method='nm')
        # correlation
        correlation_matrix = df[['P104_02', 'P104_03', 'P104_04']].corr()
        print(correlation_matrix)
        "TEST LINEARITY OF LOGIT (only look at interaction terms)"
        summary = model_with_interactions.summary()
        simple_table = summary.tables[1]
        table_data = simple_table.data
        headers = table_data[0]
        rows = table_data[1:]

        report_df = pd.DataFrame(rows, columns=headers)
        report_df = report_df.apply(pd.to_numeric, errors='ignore')
        report_df.index = ["Intercept", "P104\\_02", "P104\\_03", "P104\\_04", "P104\\_02\\_logInt","P104\\_03\\_logInt","P104\\_04\\_logInt"]
        report_df = report_df[["coef", "std err", "z", "P>|z|", "[0.025", "0.975]"]]
        report_df.round(2)
        latex_table = report_df.to_latex(float_format="%.2f", label=f"tab:mlr_linearity_table_{column_name}")
        caption = f"Linearity of logit assumption check for Statement {groups_reworked[column_name]}"
        latex_table = latex_table.replace("\\end{tabular}", "\\end{tabular}\\caption{" + caption + "}")
        latex_table = latex_table.replace("\\begin{table}", "\\begin{table}[H]")
        latex_table += "\\\\"
        file_path = 'linearity_of_logit.text'
        if os.path.exists(file_path):
            with open(file_path, 'a', encoding='utf-8') as file:
                file.write(latex_table)
        else:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(latex_table)
        print(model_with_interactions.summary())
        return "filler", pd.Series([1, 1, 1])
    else:
        X = data[['P104_02', 'P104_03', 'P104_04']]
        vif = pd.DataFrame()
        vif["Variable"] = X.columns
        vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        data = data[['P104_02', 'P104_03', 'P104_04', 'P104_combined', column_name]]
        data['P104_02'] = pd.to_numeric(data['P104_02'], errors='ignore')
        data['P104_03'] = pd.to_numeric(data['P104_03'], errors='ignore')
        data['P104_04'] = pd.to_numeric(data['P104_04'], errors='ignore')
        data['P104_combined'] = data['P104_combined'].round()
        data[column_name] = data[column_name].apply(lambda x: 1 if x > 2.5 else 0)
        data[column_name].astype('category')
        data = data.reset_index(drop=True)
        formula = f"{column_name} ~ P104_02 + P104_03 + P104_04"  # change this by either ommitting P104_03 (XP_03) or P104_04 (XP_04) or deleting both and only taking P104_combined to replicate the different version mentioned in appendix section Multinomial Logistic Regression Assumptions
        model = statsmodels.formula.api.mnlogit(formula, data).fit(method='bfgs')
        summary = model.summary()
        print(summary)
        simple_table = summary.tables[1]
        table_data = simple_table.data
        headers = table_data[0]
        rows = table_data[1:]
        report_df = pd.DataFrame(rows, columns=headers)
        report_df = report_df.apply(pd.to_numeric, errors='ignore')
        df.round(3)
        report_df['B (SE)'] = report_df.apply(lambda row: f"{round(row['coef'], 2)} ({round(row['std err'], 2)})",
                                              axis=1)
        report_df['Odds Ratio'] = report_df['coef'].apply(lambda x: round(np.exp(x), 2))
        report_df = report_df.rename(columns={'[0.025': 'Lower'})
        report_df = report_df.rename(columns={'0.975]': 'Upper'})
        report_df.loc[1:,'B (SE)'] = report_df.loc[1:].apply(add_asterisks, axis=1)
        report_df.index = ["Intercept", "experience in healthcare", "familiarity with ML", "awareness of XAI"] # delete one of these row names, if you want to run the MLR model while ommitting one predictor/ using P104_combined.
        final_df = report_df[["B (SE)", "Lower", "Odds Ratio", "Upper"]]
        final_df["Lower"][0] = ""
        final_df["Odds Ratio"][0] = ""
        final_df["Upper"][0] = ""
        latex_table = final_df.to_latex(float_format="%.2f", label=f"tab:mlr_table_{column_name}")
        caption = f"Results of the MLR model for Statement {groups_reworked[column_name]}"  # .\\\\Column definitions: \\\\\\B (SE) $\\coloneq$ Estimated regression coefficient for the predictor with standard error in brackets\\cite[p.281]{{statistics_using_r}} and asterisks expressing the level of statistical significance (* $\\coloneq$ p<0.05, ** $\\coloneq$ p<0.01, *** $\\coloneq$ p<0.001),\\\\ Lower $\\coloneq$ Lower end of the confidence interval\\cite[352]{{statistics_using_r}},\\\\ Odds Ratio $\\coloneq$ exp(B)\\cite[352]{{statistics_using_r}},\\\\ Upper $\\coloneq$ Upper bound of the confidence interval\\cite[p.352]{{statistics_using_r}}."
        latex_table = latex_table.replace("\\end{tabular}", "\\end{tabular}\\caption{" + caption + "}")
        latex_table = latex_table.replace("\\begin{table}", "\\begin{table}[H]")
        return latex_table, report_df['P>|z|'][1:]  # return p-values for average later


    """
    try:
        model = OrderedModel(
            df[column_name],
            df[["P104_02", "P104_03", "P104_04"]],
            distr="logit"  # Logistic distribution
        )
    except:
        f=0
    result = model.fit(method="bfgs")
    print(result.summary())
    mnlogit_model = sm.MNLogit(
        df[column_name],
        sm.add_constant(df[["P104_02", "P104_03", "P104_04", "G101"]])
    )
    mnlogit_result = mnlogit_model.fit()
    from scipy.stats import chi2
    ll_null = result.llf 
    ll_full = mnlogit_result.llf 
    lr_stat = 2 * (ll_full - ll_null)
    df_diff = mnlogit_result.df_model - result.df_model
    p_value = chi2.sf(lr_stat, df_diff)

    print(f"LR Test Statistic {lr_stat}")
    print(f"DOF {df_diff}")
    print(f"p-value: {p_value}")
    """

def wrap_text(series, width=40):
    """
    table formatting
    :param series:
    :param width:
    :return:
    """
    return series.apply(lambda x: "\n".join(textwrap.wrap(str(x), width)))

if __name__ == "__main__":
    # read results
    #
    """
    # original filtering script for the participant data, which could not be provided in full due to privacy reasons
    file_path = os.path.join(cfg.DATA_DIR, 'survey_results.csv')
    df = pd.read_csv(file_path, encoding='utf-8')

    ### preprocessing
    # remove rejected:
    rejected = []   # orignially contained the ids of prolific participants, which needed to be deleted for this submission
    df = df[~df['P109_01'].isin(rejected)]

    df = df[df['FINISHED'] == "1"]  # only use finished questionnaires

    failed_EQ_check = df['EQ03_04'] != "5"  # search for failed attention checks
    failed_BQ_check = df['BQ01_05'] != "5"

    # deactivated: There was only one participant who failed these checks, but he did not fail them due to inattention. At the end of the questionnaire, he mentioned trouble with interpreting the word 'rightmost'. Instead of clicking the option at the right (strongly agree), he chose agree as he though rightmost referred to the 'most right' as in correct answer.
    # eq_columns = [col for col in df.columns if col.startswith('EQ')]    # filter out answers in groups with failed attention checks
    # df.loc[failed_EQ_check, eq_columns] = None

    # bq_columns = [col for col in df.columns if col.startswith('BQ')]
    # df.loc[failed_BQ_check, bq_columns] = None
    df.drop(columns='P109_01')
    df.to_csv(os.path.join(cfg.DATA_DIR, 'filtered_survey_results.csv'), encoding='utf-8')
"""
    file_path = os.path.join(cfg.DATA_DIR, 'filtered_survey_results.csv')
    df = pd.read_csv(file_path, encoding='utf-8')

    columns_to_convert = ['BQ01_01', 'BQ01_02', 'BQ01_03', 'BQ01_04', 'BQ01_05', 'BQ02_01', 'BQ02_02', 'BQ02_03', 'BQ02_04',
                          'BQ02_05', 'BQ03_01', 'BQ03_02', 'BQ03_03', 'BQ03_04', 'BQ03_05', 'EQ01_01', 'EQ01_02', 'EQ01_03',
                          'EQ01_04', 'EQ01_05', 'EQ02_01', 'EQ02_02', 'EQ02_03', 'EQ02_04', 'EQ02_05', 'EQ03_01', 'EQ03_02',
                          'EQ03_03', 'EQ03_04', 'EQ03_05', 'GQ01_01', 'GQ01_02', 'GQ01_03', 'GQ01_04', 'GQ02_01', 'GQ02_02',
                          'GQ02_03', 'GQ02_04', 'GQ03_01', 'GQ03_02', 'GQ03_03', 'GQ03_04', 'P101', 'P102', 'P103',
                          'P104_01', 'P104_02', 'P104_03', 'P104_04', 'F101_01', 'F101_02', 'F101_03', 'F101_04', 'F101_05']
    df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')
    df['P104_combined'] = df[['P104_03', 'P104_04']].mean(axis=1)

    file_path = os.path.join(cfg.DATA_DIR, 'prolific_demographic_data.csv')
    """
    # again not provided due to private ids and data of participants
    prolific_df = pd.read_csv(file_path, encoding='utf-8')
    #prolific_df = prolific_df.drop(79, inplace=False)
    filtered_prolific_df = prolific_df[prolific_df['Participant id'].isin(df['P109_01'])]
    mean_age_rating = pd.to_numeric(filtered_prolific_df['Age']).mean()
    print(f"Mean age: {mean_age_rating:.2f}")
    std_rating = pd.to_numeric(filtered_prolific_df['Age']).std()
    print(f"Standard deviation of age: {std_rating:.2f}")
    prolific_df['Time taken'].plot.hist(bins=70, alpha=0.7, color='blue')
    std_dev = prolific_df['Time taken'].std()
    # plt.show()
    plt.close()
    """
    #drop attention check columns
    columns_to_drop = ['BQ01_05', 'BQ02_05', 'BQ03_05', 'EQ01_04', 'EQ02_04', 'EQ03_04']
    df = df.drop(columns=columns_to_drop)

    ################ General Data #################
    # balance between groups
    print('Number of participants in each group:')
    print(df['G101'].value_counts())

    # Histogram of gender, age group, educational status
    fontsize = 14
    sns.histplot(
        df['P101'],
        bins=[1, 2, 3, 4],
        kde=False,
        color='lightsteelblue',
        edgecolor='black',
        shrink=0.9
    )
    plt.title('Gender of participants', fontsize=fontsize + 2)
    plt.ylabel('Number of participants', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(ticks=[1.5, 2.5, 3.5], labels=['Female', 'Male', 'Divers'], rotation=60, fontsize=14)
    plt.subplots_adjust(bottom=0.25)
    plt.xlabel('Gender', labelpad=10, fontsize=14)
    plt.savefig(os.path.join(cfg.HISTOGRAMS, 'P101.png'))
    plt.show()
    plt.close('all')
    sns.histplot(
        df['P102'],
        bins=[1, 2, 3, 4, 5],
        kde=False,
        color='lightsteelblue',
        edgecolor='black',
        shrink=0.9
    )
    plt.title('Age group of participants', fontsize=fontsize + 2)
    plt.ylabel('Number of participants', fontsize=fontsize)
    plt.xlabel('Age Group', fontsize=fontsize)
    plt.xticks(ticks=[1.5, 2.5, 3.5, 4.5, 5.5], labels=['18-25', '26-40', '41-60', '61 and above', 'Rather not disclose'],
               rotation=60)
    plt.subplots_adjust(bottom=0.35)
    plt.savefig(os.path.join(cfg.HISTOGRAMS, 'P102.png'))
    plt.show()
    plt.close('all')
    plt.figure(figsize=(6, 8))
    sns.histplot(
        df['P103'],
        bins=[1, 2, 3, 4, 5, 6, 7, 8],
        kde=False,
        color='lightsteelblue',
        edgecolor='black',
        shrink=0.9,
    )
    plt.ylim(0, df['P103'].value_counts().max() * 2)
    plt.xlim()
    plt.title('Educational status of participants', fontsize=fontsize + 2)
    plt.ylabel('Number of participants')
    plt.xlabel('Educational Status', labelpad=10, fontsize=fontsize)
    plt.xlabel('Group', labelpad=10, fontsize=fontsize)
    plt.xticks(ticks=[1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
               labels=['General School Leaving Certificate', 'Intermediate School Certificate',
                       'General Higher Education Entrance Qualification', 'Apprenticeship', 'Bachelors Degree',
                       'Masters Degree', 'Diploma', 'Doctorate or above'], rotation=90)
    plt.subplots_adjust(bottom=0.5)
    plt.savefig(os.path.join(cfg.HISTOGRAMS, 'P103.png'))
    plt.show()
    plt.close('all')
    # user feedback to latex
    Group1 = df[df['G101'] == 1][['F102_01', 'F102_02', 'F102_03']]
    Group2 = df[df['G101'] == 2][['F102_01', 'F102_02', 'F102_03']]
    Group1 = Group1.apply(wrap_text, width=40)
    Group2 = Group2.apply(wrap_text, width=40)
    Group1_latex = Group1.to_latex(index=False, longtable=True, header=True,  column_format='|p{4cm}|p{4cm}|p{4cm}|')
    Group2_latex = Group2.to_latex(index=False, longtable=True, header=True, column_format='|p{4cm}|p{4cm}|p{4cm}|')
    #### overall mean values per group
    prefixes = ['GQ', 'EQ', 'BQ', 'F101']
    names = ['Graph Questions', 'Explanation Questions', 'Both Questions', 'Final Questions']
    likert_group_averages = pd.DataFrame()
    group1 = df[df['G101'] == 1]
    group2 = df[df['G101'] == 2]
    group1_means, group2_means = [], []
    for i, prefix in enumerate(prefixes):
        group1_means.append(group1.filter(like=prefix, axis=1).mean(axis=1).mean())
        group2_means.append(group2.filter(like=prefix, axis=1).mean(axis=1).mean())

    likert_group_averages['This'] = group1_means
    likert_group_averages['DDxT'] = group2_means

    average_group_ratings = pd.DataFrame()
    group1 = df[df['G101'] == 1]
    group2 = df[df['G101'] == 2]
    for i, prefix in enumerate(prefixes):
        mean1 = group1.filter(like=prefix, axis=1).mean(axis=1).mean()
        std1 = group1.filter(like=prefix, axis=1).mean(axis=1).std()
        mean2 = group2.filter(like=prefix, axis=1).mean(axis=1).mean()
        std2 = group2.filter(like=prefix, axis=1).mean(axis=1).std()

        average_group_ratings[f'{names[i]}'] = [
            f"{mean2:.2f} ({std2:.2f})",  # Mean and std for group2#
            f"{mean1:.2f} ({std1:.2f})"  # Mean and std for group1
        ]
    average_group_ratings = average_group_ratings.round(2)
    average_group_ratings.reset_index(drop=True)
    average_group_ratings.index = ['DDxT', 'This']
    average_group_latex = average_group_ratings.to_latex(label="tab:likert_group_averages",
                                                               float_format="%.2f")
    average_group_latex = average_group_latex.replace("\\end{tabular}",
                                                            "\\end{tabular}\\caption{Mean 5-point Likert scale ratings per model version with standard deviation in brackets over the statement groups: .\\\\ Values: 1 $\\coloneq$ `strongly disagree', 2 $\\coloneq$ `disagree', 3 $\\coloneq$ `neutral', 4 $\\coloneq$ `agree', 5 $\\coloneq$ `strongly agree'}\\\\")
    average_group_latex = average_group_latex.replace("\\begin{table}", "\\begin{table}[H]")

    print(average_group_latex)

    ############# test independence of statements between groups ###################
    print('Independence between groups:')
    for cov in ['P104_02', 'P104_03', 'P104_04']:
        group1 = df[df['G101'] == 1][cov]
        group2 = df[df['G101'] == 2][cov]
        stat, p = ranksums(group1, group2, nan_policy='omit')
        print(p)

    average_over_pages = [
        ['GQ01_01', 'GQ02_01', 'GQ03_01'],
        ['GQ01_02', 'GQ02_02', 'GQ03_02'],
        ['GQ01_03', 'GQ02_03', 'GQ03_03'],
        ['GQ01_04', 'GQ02_04', 'GQ03_04'],
        ['BQ01_01', 'BQ02_01', 'BQ03_01'],
        ['BQ01_02', 'BQ02_02', 'BQ03_02'],
        ['BQ01_03', 'BQ02_03', 'BQ03_03'],
        ['BQ01_04', 'BQ02_04', 'BQ03_04'],
        #['BQ01_05', 'BQ02_05', 'BQ03_05'],
        ['EQ01_01', 'EQ02_01', 'EQ03_01'],
        ['EQ01_02', 'EQ02_02', 'EQ03_02'],
        ['EQ01_03', 'EQ02_03', 'EQ03_03'],
        #['EQ01_04', 'EQ02_04', 'EQ03_04'],
        ['EQ01_05', 'EQ02_05', 'EQ03_05'],
    ]

    mean_names = []
    # iterate over means
    items = list(groups_reworked.items())
    for i, columns in enumerate(average_over_pages):
        name = items[i + 4][0]  # + '_mean'
        mean_names.append(name)
        df = add_average(df, columns, name)

    tables = ""
    p_df = pd.DataFrame()
    wilc_p_df = pd.DataFrame()
    for key in groups_reworked.keys():
        # if 'P' in key or 'F' in key:
        p_val = wilcoxon_ranksum(df, key)
        wilc_p_df[key] = [p_val]
        if not 'P' in key:
            report, p_vals = multinomial_log_reg(df, key, test_assumptions=False)
            p_df[key] = p_vals.rename(key)
            tables += (report + "\\\\")

    # averaged p_values over groups
    # groups G:graph, E: explanation, B: both, F: Final
    prefixes = ['G', 'E', 'B', 'F']
    groups = [1, 2, 3, 4]
    for i, prefix in enumerate(prefixes):
        a = p_df.filter(like=prefix, axis=1)
        p_df[f'Group {groups[i]} average'] = p_df.filter(like=prefix, axis=1).mean(axis=1)
        wilc_p_df[f'Group {groups[i]} average'] = wilc_p_df.filter(like=prefix, axis=1).mean(axis=1)

    s = wilc_p_df[['Group 1 average', 'Group 2 average', 'Group 3 average', 'Group 4 average']].to_latex(
        label="tab:wilcoxon_group_average", float_format="%.2f")
    average_df = p_df[['Group 1 average', 'Group 2 average', 'Group 3 average', 'Group 4 average']]
    average_df.reset_index(drop=True)
    average_df.index = ['XP\_02', 'XP\_03', 'XP\_04']
    average_df = average_df.round(2)
    print("Average p-values per group of statements:")
    print(average_df)
    average_df_latex = average_df.to_latex(label="tab:mlr_group_averages", float_format="%.2f")
    average_df_latex = average_df_latex.replace("\\end{tabular}",
                                                "\\end{tabular}\\caption{Average $p$-values over statement groups.\\\\ Group 1 $\\coloneq$ statements referring to the graph\\\\Group 2 $\\coloneq$ statements referring to the explanation\\\\Group 1 $\\coloneq$ statements referring to both, graph and explanation\\\\Group 4 $\\coloneq$ statements from the post-survey questionnaire (IDs starting with `F')\\\\}")
    average_df_latex = average_df_latex.replace("\\begin{table}", "\\begin{table}[H]")

    # averaged p_values over categories
    # categories: U: understanding, H:helpfulness, S: satisfaction, T:trust
    u_sig, h_sig, s_sig, t_sig = [], [], [], []
    substrings = ['U', 'H', 'S', 'T']
    categories = ['Understanding average', 'Helpfulness average', 'Satisfaction average', 'Trust average']
    u = ["GQ01_mean", "GQ02_mean", "GQ03_mean", "EQ01_mean", "EQ02_mean", "BQ03_mean", "F101_01", "F101_02", "F101_03"]
    h = ["GQ04_mean", "EQ03_mean", "EQ05_mean", "BQ02_mean", "F101_04"]
    s = ["BQ01_mean"]
    t = ["BQ04_mean", "F101_05"]
    substrings = [u, h, s, t]
    for i, substring in enumerate(substrings):
        p_df[f'{categories[i]}'] = p_df[substrings[i]].mean(axis=1)
        wilc_p_df[f'{categories[i]}'] = wilc_p_df[substrings[i]].mean(axis=1)

    # average likert scale ratings per category
    average_category_ratings = pd.DataFrame()
    group1 = df[df['G101'] == 1]
    group2 = df[df['G101'] == 2]
    for i, substring in enumerate(substrings):
        mean1 = group1[substrings[i]].mean(axis=1).mean()
        std1 = group1[substrings[i]].mean(axis=1).std()
        mean2 = group2[substrings[i]].mean(axis=1).mean()
        std2 = group2[substrings[i]].mean(axis=1).std()

        average_category_ratings[f'{categories[i]}'] = [
            f"{mean2:.2f} ({std2:.2f})" , # Mean and std for group2#
            f"{mean1:.2f} ({std1:.2f})"  # Mean and std for group1
        ]
    average_category_ratings = average_category_ratings.round(2)
    average_category_ratings.reset_index(drop=True)
    average_category_ratings.index = ['DDxT', 'This']
    average_category_latex = average_category_ratings.to_latex(label="tab:likert_category_averages", float_format="%.2f")
    average_category_latex = average_category_latex.replace("\\end{tabular}",
                                                "\\end{tabular}\\caption{Mean 5-point Likert scale ratings per group with standard deviation in brackets over the statement categories: Understanding, Helpfulness, Satisfaction, and Trust.\\\\ Values: 1 $\\coloneq$ `strongly disagree', 2 $\\coloneq$ `disagree', 3 $\\coloneq$ `neutral', 4 $\\coloneq$ `agree', 5 $\\coloneq$ `strongly agree'}\\\\")
    average_category_latex = average_category_latex.replace("\\begin{table}", "\\begin{table}[H]")
    print(average_category_latex)

    aa = wilc_p_df[categories].to_latex(label="tab:wilcoxon_category_average", float_format="%.2f")
    print(aa)
    row_col_map = {row: [] for row in p_df.index}
    for col in p_df.columns:
        for row in p_df.index:
            if p_df.loc[row, col] < 0.05:
                row_col_map[row].append(groups_reworked[col])
    average_df = p_df[['Understanding average', 'Helpfulness average', 'Satisfaction average', 'Trust average']]
    average_df = average_df.round(2)
    average_df.reset_index(drop=True)
    average_df.index = ['XP\_02', 'XP\_03', 'XP\_04']
    average_df_latex = average_df.to_latex(label="tab:mlr_category_averages", float_format="%.2f")
    average_df_latex = average_df_latex.replace("\\end{tabular}",
                                                "\\end{tabular}\\caption{Average $p$-values over the statement categories: Understanding, Helpfulness, Satisfaction, and Trust.}\\\\")
    average_df_latex = average_df_latex.replace("\\begin{table}", "\\begin{table}[H]")
    print(average_df_latex)



statement_codes = {
    # "P101": "What is your gender?",
    # "P102": "Choose your age group.",
    # "P103": "Choose your highest educational status.",
    "P104_01": "I am proficient in the English language.",
    "P104_02": "I am experienced in the medical field/healthcare.",
    "P104_03": "I am familiar with machine learning.",
    "P104_04": "I have heard about explainability in machine learning before.",
    "GQ01_01": "The graphs provide an intuitive understanding of how different antecedents and symptoms contribute to the diagnoses.",
    "GQ01_02": "The colored nodes in the graph allow easy identification of key symptoms and factors for the diagnoses.",
    "GQ01_03": "The coloring indicating the importance of evidences helps to understand the diagnoses.",
    "GQ01_04": "The graphs are helpful in terms of understanding the diagnoses.",
    "GQ02_01": "The graphs provide an intuitive understanding of how different antecedents and symptoms contribute to the diagnoses.",
    "GQ02_02": "The colored nodes in the graph allow easy identification of key symptoms and factors for the diagnoses.",
    "GQ02_03": "The coloring indicating the importance of evidences helps to understand the diagnoses.",
    "GQ02_04": "The graphs are helpful in terms of understanding the diagnoses.",
    "GQ03_01": "The graphs provide an intuitive understanding of how different antecedents and symptoms contribute to the diagnoses.",
    "GQ03_02": "The colored nodes in the graph allow easy identification of key symptoms and factors for the diagnoses.",
    "GQ03_03": "The coloring indicating the importance of evidences helps to understand the diagnoses.",
    "GQ03_04": "The graphs are helpful in terms of understanding the diagnoses.",
    "EQ01_01": "The explanations are comprehensible.",
    "EQ01_02": "The explanations clearly distinguish between supporting and contradicting evidences for the diagnoses.",
    "EQ01_03": "Highlighting particularly important evidences is helpful in terms of understanding the diagnoses.",
    "EQ01_04": "This is an attention check. Please select the rightmost option.",
    "EQ01_05": "The charts showing the distribution of sexes and age groups among the affected population colored by their average importance for the corresponding diagnosis are helpful.",
    "EQ02_01": "The explanations are comprehensible.",
    "EQ02_02": "The explanations clearly distinguish between supporting and contradicting evidences for the diagnoses.",
    "EQ02_03": "Highlighting particularly important evidences is helpful in terms of understanding the diagnoses.",
    "EQ02_04": "This is an attention check. Please select the rightmost option.",
    "EQ02_05": "The charts showing the distribution of sexes and age groups among the affected population colored by their average importance for the corresponding diagnosis are helpful.",
    "EQ03_01": "The explanations are comprehensible.",
    "EQ03_02": "The explanations clearly distinguish between supporting and contradicting evidences for the diagnoses.",
    "EQ03_03": "Highlighting particularly important evidences is helpful in terms of understanding the diagnoses.",
    # "EQ03_04": "This is an attention check. Please select the rightmost option.",
    "EQ03_05": "The charts showing the distribution of sexes and age groups among the affected population colored by their average importance for the corresponding diagnosis are helpful.",
    "BQ01_01": "The whole chart including graphs and explanations for the diagnoses is satisfying.",
    "BQ01_02": "Marking the most important evidences in both, graph and text, helps to understand the diagnosis.",
    "BQ01_03": "The whole chart offers a sufficient overview over the diagnoses.",
    "BQ01_04": "I can trust the diagnoses.",
    "BQ01_05": "This is an attention check. Please select the rightmost option.",
    "BQ02_01": "The whole chart including graphs and explanations for the diagnoses is satisfying.",
    "BQ02_02": "Marking the most important evidences in both, graph and text, helps to understand the diagnosis.",
    "BQ02_03": "The whole chart offers a sufficient overview over the diagnoses.",
    "BQ02_04": "I can trust the diagnoses.",
    # "BQ02_05": "This is an attention check. Please select the rightmost option.",
    "BQ03_01": "The whole chart including graphs and explanations for the diagnoses is satisfying.",
    "BQ03_02": "Marking the most important evidences in both, graph and text, helps to understand the diagnosis.",
    "BQ03_03": "The whole chart offers a sufficient overview over the diagnoses.",
    "BQ03_04": "I can trust the diagnoses.",
    "BQ03_05": "This is an attention check. Please select the rightmost option.",
    "F101_01": "My language proficiency sufficed to understand the diagnoses.",
    "F101_02": "My medical knowledge sufficed to understand the diagnoses.",
    "F101_03": "The charts included all necessary information for understanding the diagnoses.",
    "F101_04": "Overall, it was clear which evidences were important for which diagnosis.",
    "F101_05": "If I was sick, I would rely on the system providing reliable diagnoses for my disease, similar to the ones I have seen here."
}

###################### unused #####################
def one_way_anova(df, column_name, title):
    # test whether anova is applicable
    # 1. Homogeneity of Variances
    groups = [group[column_name].dropna() for _, group in df.groupby('G101')]
    stat, p = levene(*groups)
    if p < 0.05:
        print("Variances are not homogeneous: Levene's test p < 0.05")
    else:
        print("Variances are homogeneous: Levene's test p >= 0.05")

    model = ols(f'{column_name} ~ C(G101)', data=df).fit()
    residuals = model.resid
    stat, p = shapiro(residuals)
    print('Shapiro-Wilk: ', p)
    if p < 0.05:
        print(f"Not normally distributed: Shapiro-Wilk p < 0.05")
    else:
        print(f"Normally distributed: Shapiro-Wilk p >= 0.05")
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(column_name)
    print(anova_table)
    sum_sq_effect = anova_table.loc['C(G101)', 'sum_sq']
    df_effect = anova_table.loc['C(G101)', 'df']
    F_value = round(anova_table.loc['C(G101)', 'F'], 2)
    p_value = round(anova_table.loc['C(G101)', 'PR(>F)'], 2)

    sum_sq_residual = anova_table.loc['Residual', 'sum_sq']
    df_residual = anova_table.loc['Residual', 'df']
    MS_error = sum_sq_residual / df_residual
    SS_total = sum_sq_effect + sum_sq_residual

    omega_squared = round((sum_sq_effect - (df_effect * MS_error)) / (SS_total + MS_error), 2)

    anova_text = ""
    if p_value < 0.05:
        anova_text = f'There was a significant effect of group membership on the ratings on the 5-point Likert Scale:\\\\$F({df_effect},{df_residual})={F_value}$\\\\$p<.05$\\\\ $\omega={omega_squared}$'
    else:
        anova_text = f'There was no significant effect of group membership on the ratings on the 5-point Likert Scale:\\\\$F({df_effect},{df_residual})={F_value}$\\\\ $p>.05\\\\ $\omega={omega_squared}$'
    file_path = os.path.join(cfg.ANOVAS_TEXTS, column_name + '.txt')
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(anova_text)

    ### plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='G101', y=column_name, data=df, palette='pastel', hue=None, linewidth=2.5)
    sns.stripplot(x='G101', y=column_name, data=df, color='black', alpha=0.6, jitter=True, size=6)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlabel('Group', labelpad=10, fontsize=26)
    plt.ylabel('5-point Likert Scale', labelpad=10, fontsize=26)
    plt.subplots_adjust(bottom=0.16)
    plt.savefig(os.path.join(cfg.ANOVAS, column_name + '.png'))
    plt.show()


def spearmans_corr(df):
    """
    Function for calculating spearman's correlation.
    :param df:dataframe with survey results
    :return:
    """
    for cov in ['P104_02', 'P104_03', 'P104_04']:
        correlation, p_value = spearmanr(df['P104_02'], df['G101'])

        print(f"Spearman Correlation Coefficient of {cov}:", correlation)
        print("p-value:", p_value)

        if p_value < 0.05:
            print("Variables are NOT independent.")
        else:
            print("Variables are independent.")


def multiple_linear_regression(df, column_name):
    """
    # regular ancova yielded severely non-normal residuals, which can be seen by uncommenting this paragraph and looking
    at the qqplots and p_values of the shapiro-wilk test
    model_ancova = ols(f"{column_name} ~ C(G101) + P104_02 + P104_03 + P104_04", data=df).fit()
    ancova_table = sm.stats.anova_lm(model_ancova, typ=2)
    residuals = model_ancova.resid
    # plt.plot(residuals)
    shap, p = shapiro(residuals)
    statsmodels.api.qqplot(residuals, line='s')
    plt.show()
    print('shapiro: ',p)
    """
    import sys
    print(sys.getdefaultencoding())

    Group1 = df[df['G101'] == 1]  # Subset where G101 is "1"
    Group2 = df[df['G101'] == 2]  # Subset where G101 is "2"
    rows_to_drop = df[df['G101'] == 1].index[:2]
    data = df.drop(rows_to_drop)
    data = data[['G101', 'P104_02', 'P104_03', 'P104_04', column_name]]
    data = data.reset_index(drop=True)

    pandas2ri.activate()

    ro.r('Sys.setlocale("LC_ALL", "en_US.UTF-8")')
    utils = importr('utils')
    base = importr('base')
    grdevices = importr('grDevices')
    utils.chooseCRANmirror(ind=1)

    ro.globalenv['column_name'] = column_name

    with (ro.default_converter + pandas2ri.converter).context():
        r_from_pd_df = ro.conversion.get_conversion().py2rpy(data)
    ro.globalenv['df'] = r_from_pd_df

    ro.r('''
        df <- na.omit(df)
        #df$G101 <- as.factor(df$G101)
        #install.packages("car")
        #install.packages("lmtest")
        #install.packages("QuantPsyc")
        library(WRS2)
        library(boot)
        library(car)
        library(lmtest)
        library(QuantPsyc)
        formula <- as.formula(paste(column_name, "~ G101 + P104_02 + P104_03 + P104_04"))

        #formula <- as.formula(paste(column_name, "~", covariate, "+ G101")) # old, for ancboot
        #result <- ancboot(formula, data = df)  # bootstrapped ancova. Only works for one covariate at most, thus was replaced by lm
        #print(summary(result$p.vals))
        #out_p <- summary(result$p.vals)
        #out_trDiff <- summary(result$trDiff)
        #out_cilow <- summary(result$ci.low)
        #out_cihi <- summary(result$ci.hi)

        print('Linear regression:')

        lr_result = lm(formula, data = df)
        # test assumptions for linear regression, set to true if you want to run them
        if(FALSE){
            # 1. Variable Types:print structure of the data
            cat("Variable Types and Summary:\n")
            str(df)
            cat("\nOutcome Variable Summary:\n")
            print(summary(df$outcome_variable))

            # 2. Non-Zero Variance: print variances of predictors
            cat("\nVariance of Predictor Variables:\n")
            predictor_vars <- colnames(df)[colnames(df) != "outcome_variable"]
            variances <- apply(df[, predictor_vars], 2, var)
            print(variances)

            # 3.No Perfect Multicollinearity:  Print VIFs
            cat("\nVariance Inflation Factors (VIF):\n")
            vif_values <- vif(lr_result)
            print(vif_values)

            # 4. Homoscedasticity: Plot residuals vs fitted values
            png("residuals_vs_fitted.png", width = 800, height = 600)
            plot(lr_result$fitted.values, residuals(lr_result),
                 main = "Residuals vs Fitted",
                 xlab = "Fitted Values",
                 ylab = "Residuals")
            abline(h = 0, col = "red")
            dev.off()

            # 5.Independent Errors: Durbin-Watson test
            cat("\nDurbin-Watson Test for Independent Errors:\n")
            dw_test <- dwtest(lr_result)
            print(dw_test)

            # 6. Normally distributed errors: QQ-plot and Shapiro-Wilk Test
            png("qqplot.png", width = 800, height = 600)
            qqnorm(residuals(lr_result), main = "Q-Q Plot of Residuals")
            qqline(residuals(lr_result), col = "red", lwd = 2)
            dev.off()

            cat("\nShapiro-Wilk Test for normality of residuals\n")
            shapiro_test <- shapiro.test(residuals(lr_result))
            print(shapiro_test)

            rainbow_result <- raintest(lr_result)
            print(rainbow_result) 

            # qq plots to check non normality
            png(file="qqplot.png", width=800, height=600)
            qqnorm(resid(lr_result))
            qqline(resid(lr_result), col="red", lwd=2)
            dev.off()
        }
        # For the survey data of this project every assumption was met, except normal distribution (for most variables, for some it was ok)
        # -> bootstrapping to look at confidence intervals

        summary_result <- summary(lr_result)

        residual_se <- summary_result$sigma
        r_squared <- summary_result$r.squared
        adj_r_squared <- summary_result$adj.r.squared
        f_statistic <- summary_result$fstatistic[1]
        f_df1 <- summary_result$fstatistic[2]
        f_df2 <- summary_result$fstatistic[3]
        f_p_value <- pf(f_statistic, f_df1, f_df2, lower.tail = FALSE)

        summary_values <- list(
            residual_se = residual_se,
            r_squared = r_squared,
            adj_r_squared = adj_r_squared,
            f_statistic = f_statistic,
            f_df1 = f_df1,
            f_df2 = f_df2,
            f_p_value = f_p_value
        )

        print(summary_result)
        print("confidence interval:")   
        conf_int = confint(lr_result)
        print(conf_int)

        output_df <- data.frame(
            Estimate = coef(summary_result)[, "Estimate"],
            StdError = coef(summary_result)[, "Std. Error"],
            tValue = coef(summary_result)[, "t value"],
            Pr = coef(summary_result)[, "Pr(>|t|)"],
            CI_Lower = conf_int[, 1],
            CI_Upper = conf_int[, 2]
        )
        print(output_df)
        set.seed(123)
        n <- 100
        boot_function <- function(df, indices) {
          boot_data <- df[indices, ]
          model <- lm(formula, data = boot_data)
          return(coef(model))
        }

        set.seed(123)
        boot_results <- boot(
          data = df,
          statistic = boot_function,
          R = 1000 #  #bootstrap samples
        )

        boot0 <- boot.ci(boot_results, index = 2)
        boot1 <- boot.ci(boot_results, index = 3)
        boot2 <- boot.ci(boot_results, index = 4)

        basic_formula <- as.formula(paste(column_name, "~ G101"))

        model1 <- lm(formula = basic_formula, data = df)

        model2 <- lm(formula = formula, data = df)

        model2_confint <- confint(model2)

        R2_step1 <- summary(model1)$r.squared
        R2_step2 <- summary(model2)$r.squared

        delta_R2 <- R2_step2 - R2_step1

        cat("Step 1 R²:", R2_step1, "\n")
        cat("Step 2 R²:", delta_R2, "\n")

        model2_summary <- summary(model2)
        model1_summary <- summary(model1)

        coefficients <- model2_summary$coefficients
        model1.beta <- lm.beta(model1)
        model2.beta <- lm.beta(model2)

        coef_table2 <- data.frame(
          Predictor = rownames(coefficients),
          B = coefficients[, "Estimate"],
          SE_B = coefficients[, "Std. Error"],
          p = coefficients[, "Pr(>|t|)"]
        )

        coefficients1 <- model1_summary$coefficients
        coef_table1 <- data.frame(
          Predictor = rownames(coefficients1),
          B = coefficients1[, "Estimate"],
          SE_B = coefficients1[, "Std. Error"],
          p = coefficients1[, "Pr(>|t|)"]
        )

        report <- data.frame(
          Step = c("1", "2"),
          R2 = c(R2_step1, delta_R2)
        )

    ''')
    # if you want to look at qq plots, set the if(FALSE) in the r above to TRUE and uncomment the following
    """
    from PIL import Image
    img = Image.open("qqplot.png")
    plt.imshow(img)
    plt.axis('off')  # Turn off axes
    plt.show()
    """
    with (ro.default_converter + pandas2ri.converter).context():
        r_coefficients1 = ro.globalenv['coefficients1']
        r_coefficients2 = ro.globalenv['coefficients']
        r_model1_beta = ro.globalenv['model1.beta']
        r_model2_beta = ro.globalenv['model2.beta']
        r_report = ro.globalenv['report']
        r_model2_confint = ro.globalenv['model2_confint']
        r_boot0 = ro.globalenv['boot0']
        r_boot1 = ro.globalenv['boot1']
        r_boot2 = ro.globalenv['boot2']
        model_2_confint = ro.conversion.get_conversion().rpy2py(r_model2_confint)
        boot0 = ro.conversion.get_conversion().rpy2py(r_boot0)
        boot1 = ro.conversion.get_conversion().rpy2py(r_boot1)
        boot2 = ro.conversion.get_conversion().rpy2py(r_boot2)
        coefficients1 = ro.conversion.get_conversion().rpy2py(r_coefficients1)
        coefficients2 = ro.conversion.get_conversion().rpy2py(r_coefficients2)
        model1_beta = ro.conversion.get_conversion().rpy2py(r_model1_beta)
        model2_beta = ro.conversion.get_conversion().rpy2py(r_model2_beta)
        report = ro.conversion.get_conversion().rpy2py(r_report)

    m1_betas = [None]
    m2_betas = [None]
    m1_betas.extend(model1_beta)
    m2_betas.extend(model2_beta)
    boot0_interval = boot0['bca'][0][3:5]
    boot1_interval = boot1['bca'][0][3:5]
    boot2_interval = boot2['bca'][0][3:5]
    result_table = []
    for i, r_sq in enumerate(report["R2"]):
        result_table.append([r_sq, None, None, None, None])
        if i == 0:
            for ri, row in enumerate(coefficients1):
                print(m1_betas[ri])
                result_table.append([None, row[0], row[1], m1_betas[ri], row[2]])
        elif i == 1:
            for ri, row in enumerate(coefficients2):
                print(m2_betas[ri])
                result_table.append([None, row[0], row[1], m2_betas[ri], row[2]])
    row_names = ["Step 1", "Constant", "Group", "Step 2", "Constant", "Group", groups["P104_02"], groups["P104_03"],
                 groups["P104_04"]]
    columns = ["ΔR²", "B", "SE B", "\\beta", "P"]
    lm_model_df = pd.DataFrame(result_table, columns=columns)
    lm_model_df.index = row_names
    lm_model_df = lm_model_df.applymap(round_and_format)
    lm_model_df["95\\% CI"] = [None, None, None, None, None, None,
                               [round(boot0_interval[0], 2), round(boot0_interval[1], 2)],
                               [round(boot1_interval[0], 2), round(boot1_interval[1], 2)],
                               [round(boot2_interval[0], 2), round(boot2_interval[1], 2)]]
    #### text results related to Rsquared
    m1_r2 = lm_model_df.iloc[0]["ΔR²"]
    if '<' in str(m1_r2):
        m1_r2_number = float(m1_r2[1:]) * 100
        m1_r2 = '<' + str(m1_r2_number)
    else:
        m1_r2 = float(m1_r2) * 100
        m1_r2_number = m1_r2
    beneficial = False
    m2_r2 = lm_model_df.iloc[3]["ΔR²"]
    if '<' in str(m2_r2):
        m2_r2 = float(m2_r2[1:]) * 100
        if m2_r2 > m1_r2_number:
            beneficial = True
        m2_r2 = '<' + str(m2_r2)
    else:
        m2_r2 = float(m2_r2) * 100
        if m2_r2 > m1_r2_number:
            beneficial = True
    text = f"The group membership of a participant accounts for {m1_r2}\\% of the variation of the ratings of statement {groups[column_name]}."
    if beneficial:
        text = f"The inclusion of the covariates as predictors explained {m2_r2}\\% more of the variation of the ratings to statement {groups[column_name]}."
    else:
        text += f"The inclusion of the covariates did not explain more of the variation of the ratings of statement {groups[column_name]}."
    text += "\\\\"
    no_effect_covariates = "The following covariates do not contribute significantly ($p>5\%$):"
    for i, p in enumerate(lm_model_df["P"][6:], start=6):
        try:
            na = math.isnan(p)
        except:
            na = False
        if p != None and not na:
            if '<' in str(p):
                p_value = float(str(p)[1:]) * 100
            else:
                p_value = float(p) * 100
            if p_value < 5:
                text += f"The p-value {p_value}\\% is below 5\\%, which indicates a significant contribution of covariate {row_names[i]} to the model.\\\\"

                beta = lm_model_df["\\beta"][i]
                B = lm_model_df["B"][i]
                if '<' in str(B):
                    B = float(str(B)[1:]) * 100
                else:
                    B = float(p) * 100
                if B > 0:
                    text += f"Since B is bigger than 0, the relationship between the ratings on statement {groups[column_name]} and covariate {row_names[i]} is positive. "
                    text += f"As the rating of covariate {row_names[i]} increases by one standard deviation, the rating of statement {groups[column_name]} increases by standardized B-value $\\beta={beta}$."
                else:
                    text += f"Since B is smaller than 0, the relationship between the ratings on statement {groups[column_name]} and covariate {row_names[i]} is negative."
                    text += f"As the rating of covariate {row_names[i]} increases by one standard deviation, the rating of statement {groups[column_name]} decreases by standardized B-value $\\beta={beta}$."
                if i > 5:
                    CI = lm_model_df["95\\% CI"][i]
                    text += "\\\\"
                    if (CI[0] < 0 < CI[1]) or (CI[0] > 0 > CI[1]):
                        text += f"The confidence interval contains zero, which indicates that the predictor exhibits both, samples with positive and negative relationships to the outcome.\\\\"
                    elif CI[0] > 0 and CI[1] > 0:
                        text += "The confidence interval does not contain zero and both numbers are positive, which suggests a consistent positive relationship between predictor and outcome across samples.\\\\"
                    else:
                        text += "The confidence interval does not contain zero and both numbers are negative, which suggests a consistent negative relationship between predictor and outcome across samples.\\\\"
            else:
                no_effect_covariates += row_names[i] + ", "
                # text += f"The covariate {row_names[i]} does not contribute significantly to the model $({round(float(p)*100, 2)}% > 5%)$."
    no_effect_covariates = no_effect_covariates[:-2] + "."
    text += no_effect_covariates
    lm_model_df = lm_model_df.fillna("")
    latex_table = lm_model_df.to_latex(float_format="%.2f", label=f"tab:mlr_table_{column_name}")
    caption = f"Results of the MLR model Step 1 and Step 2 formulas for Statement {groups[column_name]}.\\\\Column definitions: \\\\\\Delta R^2 $\\coloneq$ measure for the amount of variability in the outcome that is accounted for by the predictors\\cite[p.281]{{statistics_using_r}},\\\\ B $\\coloneq$ Estimate for B-values from \\autoref{{eq:step2}}\\cite[282]{{statistics_using_r}},\\\\ SE B $\\coloneq$ Standard Error\\cite[280]{{statistics_using_r}},\\\\ \\beta $\\coloneq$ Estimate B, measured in standard deviation units, which gives the number of standard deviations by which the outcome will change as a result of one standard deviation change in the predictor\\cite[p.283]{{statistics_using_r}},\\\\ P $\\coloneq$ Result of t-test associated with B-value\\cite[p.282]{{statistics_using_r}},\\\\ 95\\% CI $\\coloneq$ For 95\\% of bootstrapped samples, this interval contains the true value of \\beta\\cite[p.284]{{statistics_using_r}}."

    latex_table = latex_table.replace("\\end{tabular}", "\\end{tabular}\\caption{" + caption + "}")
    latex_table = latex_table.replace("\\begin{table}", "\\begin{table}[H]")

    title = f"Multiple Linear Regression for the three covariates (5: experience in healthcare, 6: familiarity with AI, and 7: awareness of XAI) and Statement `{groups[column_name]}: {statement_codes[column_name]}'\\\\"
    final_page = title + latex_table + "\\\\" + text + "\\newpage"
    # grouped_data = df.groupby(['P104_02', 'P104_03'])[f'{column_name}'].mean().reset_index()

    a = 0

    """
    # an attempt at bootstrapping in python - aborted, since I had no way to check whether the results
    # would be correct (in python). If I had to switch to R anyways, why even bother with python bootstrapping?
    formula = f"{column_name} ~ G101 + P104_02"
    observed_model = ols(formula, data=df).fit()
    observed_ancova = sm.stats.anova_lm(observed_model)
    observed_F = observed_ancova.loc['G101', 'F']
    n_bootstraps = 1000
    bootstrap_F_values = []
    bootstrap_adjusted_means = []
    for _ in range(n_bootstraps):
        bootstrap_sample = df.sample(n=len(df), replace=True)   #resampling
        model = ols(formula, data=bootstrap_sample).fit()
        anova_results = sm.stats.anova_lm(model)
        #group_F = anova_results.loc['group', 'F']
        ##bootstrap_F_values.append(group_F)
        means = bootstrap_sample.groupby('G101').apply(
            lambda group: model.predict(
                pd.DataFrame({'G101': [group.name], 'P104_02': [df['P104_02'].mean()]})
            ).iloc[0]
        )
        bootstrap_adjusted_means.append(means)

    bootstrap_adjusted_means = pd.DataFrame(bootstrap_adjusted_means)
    ci_adjusted_means = {}
    for group_name in bootstrap_adjusted_means.columns:
        ci_adjusted_means[group_name] = bootstrap_adjusted_means[group_name].quantile([0.025, 0.975])
    for group, ci in ci_adjusted_means.items():
        print(f"Group {group}: Adjusted Mean 95% CI = [{ci.iloc[0]:.3f}, {ci.iloc[1]:.3f}]")
    #group_A_ci = bootstrap_adjusted_means[0].quantile([0.025, 0.975])
    #group_B_ci = bootstrap_adjusted_means[1].quantile([0.025, 0.975])

    #bootstrap_F_values = np.array(bootstrap_F_values)
    #lower_ci = np.percentile(bootstrap_F_values, 2.5)
    #upper_ci = np.percentile(bootstrap_F_values, 97.5)
    #p_value = np.mean(bootstrap_F_values >= observed_F)
    """


# original graphs were randomized, but answer options were not. This function switches everyting back into place
# only necessary if you want to look at the results per Diagnosis shown to the participants.
# for the evaluation of this thesis, only the averages mattered, so no reordering necessary
def rearrange_responses(row):
    order = [int(row['G103x01']), int(row['G103x02']), int(row['G103x03'])]
    both_responses = {
        0: row[['BQ01_01', 'BQ01_02', 'BQ01_03', 'BQ01_04', 'BQ01_05']].tolist(),
        1: row[['BQ02_01', 'BQ02_02', 'BQ02_03', 'BQ02_04', 'BQ02_05']].tolist(),
        2: row[['BQ03_01', 'BQ03_02', 'BQ03_03', 'BQ03_04', 'BQ03_05']].tolist()
    }
    graph_responses = {
        0: row[['GQ01_01', 'GQ01_02', 'GQ01_03', 'GQ01_04']].tolist(),
        1: row[['GQ02_01', 'GQ02_02', 'GQ02_03', 'GQ02_04']].tolist(),
        2: row[['GQ03_01', 'GQ03_02', 'GQ03_03', 'GQ03_04']].tolist()
    }
    explanation_responses = {
        0: row[['EQ01_01', 'EQ01_02', 'EQ01_03', 'EQ01_04', 'EQ01_05']].tolist(),
        1: row[['EQ02_01', 'EQ02_02', 'EQ02_03', 'EQ02_04', 'EQ02_05']].tolist(),
        2: row[['EQ03_01', 'EQ03_02', 'EQ03_03', 'EQ03_04', 'EQ03_05']].tolist()
    }
    for i in range(0, 3):
        idx = order.index(i + 1)
        both_response_values = both_responses[idx]
        graph_response_values = graph_responses[idx]
        ex_response_values = explanation_responses[idx]

        row[f'BQ{i + 1:02d}_01'] = both_response_values[0]
        row[f'BQ{i + 1:02d}_02'] = both_response_values[1]
        row[f'BQ{i + 1:02d}_03'] = both_response_values[2]
        row[f'BQ{i + 1:02d}_04'] = both_response_values[3]
        row[f'BQ{i + 1:02d}_05'] = both_response_values[4]

        row[f'GQ{i + 1:02d}_01'] = graph_response_values[0]
        row[f'GQ{i + 1:02d}_02'] = graph_response_values[1]
        row[f'GQ{i + 1:02d}_03'] = graph_response_values[2]
        row[f'GQ{i + 1:02d}_04'] = graph_response_values[3]

        row[f'EQ{i + 1:02d}_01'] = ex_response_values[0]
        row[f'EQ{i + 1:02d}_02'] = ex_response_values[1]
        row[f'EQ{i + 1:02d}_03'] = ex_response_values[2]
        row[f'EQ{i + 1:02d}_04'] = ex_response_values[3]
        row[f'EQ{i + 1:02d}_05'] = ex_response_values[4]
    return row

# df = df.apply(rearrange_responses, axis=1)