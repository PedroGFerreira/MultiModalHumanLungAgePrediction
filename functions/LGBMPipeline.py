import pickle
import re
import lightgbm as lgbm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import linregress, spearmanr
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
from functions.convert_gene_name import convert_gene_name

sns.set(style="ticks", font_scale=1.05)

class LGBMPipeline(BaseEstimator, ClassifierMixin):
    def __init__(self, test_size=0.3, rand=42, n_jobs=-1,
                 params={"lambda_l1": 0.01, "lambda_l2": 0.1, "max_depth": 100},
                 model=None):
        self.test_size = test_size
        self.rand = rand
        self.n_jobs = n_jobs
        if model:
            self.model = model
        else:
            self.model = lgbm.LGBMRegressor(random_state=self.rand, n_jobs=-self.n_jobs, **params)

    def fit(self, X, y):
        self.model.fit(X, y, eval_metric=["rmse", "r2"])

    def predict(self, X):
        return self.model.predict(X)

    def compute_metrics(self, y_test, y_test_pred):
        metrics = {}
        metrics["r2"] = r2_score(y_test, y_test_pred)
        metrics["rmse"] = mean_squared_error(y_test, y_test_pred, squared=False)
        metrics["mae"] = mean_absolute_error(y_test, y_test_pred)
        metrics["med"] = median_absolute_error(y_test, y_test_pred)
        lr = linregress(y_test, y_test_pred)
        metrics["slope"] = lr.slope
        metrics["intercept"] = lr.intercept
        metrics["cor"] = lr.rvalue
        return metrics

    def plot_model_fit(self, X_test, y_test,
                       data_modality="Gene Expression", data_set="Validation",
                       smoker_status=None, color="#E87B26",
                       title_override=None,
                       fig_output_path="scatterplot.pdf"):
        y_test_pred = self.predict(X_test)
        metrics = self.compute_metrics(y_test, y_test_pred)

        # kind="reg" is not supported when using hue;
        # as a workaround we plot scatter separately.
        if smoker_status is not None:
            scatter = False
        else:
            scatter = True

        jointgrid = sns.jointplot(x=y_test, y=y_test_pred,
                                  kind="reg",
                                  truncate=False,
                                  scatter=scatter,
                                  fit_reg=True,
                                  color=color,
                                  xlim=(20, 70),
                                  ylim=(20, 70))
        jointgrid.ax_joint.axline([0, 0], [1, 1], transform=jointgrid.ax_joint.transAxes,
                                  linestyle="--", alpha=0.8, color='darkgray')

        if smoker_status is not None:
            sns.scatterplot(x=y_test, y=y_test_pred, hue=smoker_status, ax=jointgrid.ax_joint)
            sns.move_legend(jointgrid.ax_joint, "lower right")

        if title_override:
            plt.title(title_override)
        else:
            plt.title(f"LGBM Model Fit for {data_modality}")# - {data_set} (N = {X_test.shape[0]})")
        jointgrid.ax_joint.set_ylabel("Predicted Values")
        jointgrid.ax_joint.set_xlabel("True Value")
        t = plt.text(.05, .65,
                     'r²={:.2f}\nrmse={:.2f}\nmae={:.2f}\nmed={:.2f}\nslope={:.2f}\nintercept={:.2f}\ncor={:.2f}'.format(
                         metrics["r2"], metrics["rmse"], metrics["mae"], metrics["med"], metrics["slope"],
                         metrics["intercept"], metrics["cor"]),
                     transform=jointgrid.ax_joint.transAxes)
        t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor=color))
        jointgrid.fig.subplots_adjust(top=0.95)
        plt.tight_layout()
        if fig_output_path:
            plt.savefig(fig_output_path, format="pdf", bbox_inches="tight")
        plt.show()
        plt.close('all')

    def plot_distplot(self, X_test, y_test, type="Validation", color="#E87B26"):
        # Trocar y_pred e y_test
        y_test_pred = self.predict(X_test)
        sns.displot(data=(y_test - y_test_pred), kde=True, color=color)
        plt.title(f"LGBM Difference Plot {type} (N = {X_test.shape[0]})")
        plt.xlabel( r"Age: $y_{test}$ - $y_{pred}$")

    def plot_top_n_genes(self, n=10, color="#E87B26",
                         importance_type="split",
                         data_modality="Gene Expression",
                         fig_output_path="feature_importances.pdf"):
        sns.set(style="ticks", font_scale=1.2)
        # Top n (default 10) genes by importance
        ax_lgbm = lgbm.plot_importance(self.model, max_num_features=n,
                                       title=f"LGBM Top {n} Features - {data_modality}",
                                       color=color, importance_type=importance_type,
                                       xlabel=f"Feature Importance - {importance_type.capitalize()}",
                                       ylabel="", grid=False, precision=1, height=0.8,
                                       label=None)
        for txt in ax_lgbm.texts:
            txt.set_visible(False)
        ax_lgbm.figure.set_size_inches(7.5, 1.2*n)
        # Rename genes from Ensembl IDs to common gene name
        ax_lgbm.set_yticklabels(convert_gene_name(ax_lgbm.get_yticklabels()))
        ax_lgbm.set_ylim(top=4.5, bottom=-.5)

        plt.tight_layout()
        if fig_output_path:
            plt.savefig(fig_output_path, format="pdf", bbox_inches="tight", dpi=600)
        plt.show()
        plt.close('all')
        sns.set(style="ticks", font_scale=1.05)

    def get_top_n_genes(self, n=10):
        # Top n genes by importance
        ax_lgbm = lgbm.plot_importance(self.model, max_num_features=n)
        # Rename genes from Ensembl IDs to common gene name
        ax_lgbm.set_yticklabels(convert_gene_name(ax_lgbm.get_yticklabels()))
        genes = [label.get_text() for label in ax_lgbm.get_yticklabels()]
        return genes

    def get_model(self):
        return self.model

    def save_model(self, basename="booster"):
        # save with pickle
        pickle.dump(self.model, open(f"{basename}.pkl", "wb"))
        # save with lightgbm
        self.model.booster_.save_model(f"{basename}.txt")

    def compute_correlation(self, X, y, covar_filename, exclude_cols=["SUBJID", "ExpID", "MHSMKNMB", "MHSMKYRS"]):
        """Compute correlation of prediction error with covariates"""

        covars = pd.read_csv(covar_filename)
        covars = covars.set_index("histologyID")
        # Convert the ids into histology valid ids
        regex = re.compile(r"-SM-.*")
        samples_ids = [regex.sub('', s) for s in list(X.index)]

        # Subset the covar dataset
        covars_set = covars.loc[samples_ids]
        covars_set_subset = covars_set.drop(exclude_cols, axis=1)

        # One-hot encode categorical variables
        for col in covars_set_subset.dtypes[covars_set_subset.dtypes == "object"].index:
            one_hot = pd.get_dummies(col, prefix=col)
            covars_set_subset = covars_set_subset.drop(col, axis=1)
            covars_set_subset = covars_set_subset.join(one_hot)

        # Make prediction and compute error
        # select only features used by the model
        y_test_pred = self.model.predict(X.loc[:, self.model.feature_name()])
        y_error = abs(y - y_test_pred)

        # Correlation
        correlations = {}
        p_values = {}
        for col in covars_set_subset.columns:
            corr, p_val = spearmanr(covars_set_subset[col], y_error)
            correlations[col] = corr
            p_values[col] = p_val

        result_df = pd.DataFrame({
            'Spearman correlation': correlations,
            'p-value': p_values
        })

        return result_df
