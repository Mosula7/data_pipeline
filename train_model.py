import os
import sys
import json

import numpy as np
import pandas as pd

import catboost as cat
import lightgbm as lgb

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    ConfusionMatrixDisplay, classification_report, roc_curve
)
from scipy.stats import ks_2samp
from operator import itemgetter


class TrainModel:
    def __init__(self, model_type, data_dir, hyperparams):

        self.model_type = model_type
        self.data_dir = data_dir
        self.hyperparams = hyperparams
        datasetsets = (
            'X_train', 'X_valid', 'X_test',
            'y_train', 'y_valid', 'y_test'
        )
        for d in datasetsets:
            self.__setattr__(
                d,
                pd.read_csv(os.path.join(self.data_dir, f'{d}.csv'))
            )

    def train_lgb(self):

        model = lgb.LGBMClassifier(**self.hyperparams)
        model.fit(
            self.X_train, self.y_train,
            eval_set=(self.X_valid, self.y_valid)
        )

    def train_cat(self):

        model = cat.CatBoostClassifier(**self.hyperparams)
        model.fit()

    def evaluate_model(
            self,
            y,
            pred,
            title='Metrics',
            scoring_func=lambda x: 487.123 + 28.8539 * np.log((1 - x) / x),
            threshold=0.5
            ):
        """
        makes 4 plots:
        first plot is a PDF of probabilities or scores
        second plot is an roc curve
        third plot is a confusion matrix a
        fourth plot is a classification report
        the function also calculates KS statistic, AUC and accuracy

        then the function saves this plot and also return model metrics:
          KS, Accuracy, AUC
        """

        title_font_size = 14
        table_cmap = sns.cubehelix_palette(
            start=2, rot=0, dark=0.2, light=1, as_cmap=True
        )
        fig, ax = plt.subplots(2, 2, figsize=(14, 12))

        auc = roc_auc_score(y, pred)

        # KS
        ks_data = pd.DataFrame({'Target': y, 'prob': pred})
        ks_data['SCORE'] = ks_data['prob'].apply(scoring_func)
        sns.histplot(
            data=ks_data, x='SCORE', hue='Target',
            stat='probability', kde=True, bins=20,
            common_bins=False, common_norm=False,
            palette=['darkorange', 'grey'], edgecolor='black',
            ax=ax[0][0]
        )
        ax[0][0].set_title('PDF', fontsize=title_font_size)

        # ROC
        fpr, tpr, _ = roc_curve(y, pred)

        ax[0][1].plot(fpr, tpr, 'orange', label='AUC = %0.2f' % auc)
        ax[0][1].set_title('ROC')

        ax[0][1].legend(loc='lower right')
        ax[0][1].plot([0, 1], [0, 1], 'grey', linestyle='--')
        ax[0][1].set_xlim([0, 1])
        ax[0][1].set_ylim([0, 1])
        ax[0][1].set_ylabel('True Positive Rate')
        ax[0][1].set_xlabel('False Positive Rate')

        # confusion matrix
        ax[1][0].grid(None)
        ConfusionMatrixDisplay.from_predictions(
            y, pred > threshold, ax=ax[1][0], cmap=table_cmap
        )
        ax[1][0].grid(None)
        ax[1][0].set_title('Confusion Matrix', fontsize=title_font_size)

        # classification report
        cr = pd.DataFrame(itemgetter(*tuple(y.unique().astype('str')))(
            classification_report(y, pred > threshold, output_dict=True))
            ).drop(columns='support')

        sns.heatmap(cr, annot=True, vmax=1,
                    fmt='.5f', ax=ax[1][1], cmap=table_cmap
                    )
        ax[1][1].set_title('Classification Report', fontsize=title_font_size)

        ks = ks_2samp(
            ks_data.query('Target == 0')['SCORE'],
            ks_data.query('Target == 1')['SCORE']
        )[0]
        acc = accuracy_score(y, pred > .5)

        # Title
        fig.suptitle(f""" {title} \n
                    KS - {ks:.3f}\
                    AUC - {auc:.3f}\
                    Accuracy - {acc:.3f}""", fontsize=16)

        # formating
        for _, spine in ax[1][1].spines.items():
            spine.set_visible(True)

        cbar = ax[1][1].collections[0].colorbar
        cbar.outline.set_edgecolor('black')
        cbar.outline.set_linewidth(0.75)

        ax[0][0].set_axisbelow(True)
        ax[0][0].yaxis.grid(
            True, which='major', color='lightgrey',
            linestyle='-', linewidth=0.5
        )
        ax[0][1].set_axisbelow(True)
        ax[0][1].yaxis.grid(
            True, which='major', color='lightgrey',
            linestyle='-', linewidth=0.5)

        ax[0][0].spines[['left', 'right', 'top']].set_visible(False)
        ax[0][1].spines[['left', 'right', 'top']].set_visible(False)

        plt.tight_layout()
        plt.savefig('figures/plot.png', dpi=300)

        return {'KS': ks, 'Accuracy': acc, 'AUC': auc}

    def run_pipeline(self):

        self.__getattribute__(f'train_{self.model_type}')
        self.evaluate_model()


if __name__ == '__main__':
    model_type = sys.argv[1]

    with open(os.path.join('configs', f'config_{model_type}.json')) as file:
        data_dir = json.load(file)['data']['out_dir']

    with open(os.path.join('configs', f'config_{model_type}.json')) as file:
        hyperparams = json.load(file)['model']

    model_pipeline = TrainModel(
        model_type=model_type,
        data_dir=data_dir,
        hyperparams=hyperparams
    )

    model_pipeline.run_pipeline()
