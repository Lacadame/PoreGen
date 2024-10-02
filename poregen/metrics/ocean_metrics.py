import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


class Ocean_Metrics():

    """Compute the Brier score for diferent models
        x = [n_models, predictions,n_ensembles, chanels, w,h]"""
    def __init__(self):
        pass

    def Brier_score(self, x_hat, x, ssh_thresholds: dict = None,
                    velocity_thresholds: dict = None):
        """
        Calculate Brier score for predictions against actual observations.

        :param x_hat: Tensor of shape [n_models, n_samples, n_ensembles,
                                       channels, w, h]
         containing the ensemble predictions.
        :param x: Tensor of shape [n_models, n_samples, channels, w, h]
         containing the actual observations.
        :param ssh_thresholds: Dictionary of thresholds for SSH with operators
         as keys and list of thresholds as values.
        :param velocity_thresholds: Dictionary of thresholds for velocity
         components with operators as keys and list of thresholds as values.
        :return: Brier scores for velocity U, velocity V, and SSH. In the
         order: [n_thresholds, n_models, predictions] for each U,V,SSH
        """
        n_models, n_samples, n_ensembles, channels, w, h = x_hat.shape
        brier_ssh = {'<=': [], '>=': []}
        brier_u = {'<=': [], '>=': []}
        brier_v = {'<=': [], '>=': []}

        for operator, thresholds in ssh_thresholds.items():
            for tresh in thresholds:
                phat = torch.mean((x_hat <= tresh if operator == '<=' else x_hat >= tresh).float(), dim = 2)
                # phat = [n_models, predictions, chanels ,w,h
                px = (x <= tresh if operator == '<=' else x >= tresh).float()
                # px = [n_models, predictions, chanels ,w,h]
                # Calculate the Brier score

                brier = torch.sum((px[:, :, 0] - phat[:, :, 0]) ** 2, dim=(2, 3)) / (n_samples * w * h)
                # brier = [n_models, predictions]
                brier_ssh[operator].append(brier)

        for operator, thresholds in velocity_thresholds.items():
            for tresh in thresholds:
                phat = torch.mean((x_hat <= tresh if operator == '<=' else x_hat >= tresh).float(), dim = 2)
                px = (x <= tresh if operator == '<=' else x >= tresh).float()

                # Calculate the Brier score
                # phat = [n_models, predictions , n_chanels, w, h]
                brier1 = torch.mean((px[:, :, 1] - phat[:, :, 1]) ** 2,
                                    dim=(2, 3)) / (n_samples * w * h)
                brier2 = torch.mean((px[:, :, 2] - phat[:, :, 2]) ** 2,
                                    dim=(2, 3)) / (n_samples * w * h)
                # brier = [n_models, predictions, chanels -1]

                brier_u[operator].append(brier1)
                brier_v[operator].append(brier2)

        return brier_u, brier_v, brier_ssh

    def EMRMS(self, predictions, actuals):
        """
        Calculate the Ensemble Mean RMSE for each channel and the total RMSE
        across all channels.

        :param predictions: Tensor of shape [n_models, n_predictions,
                                             n_ensembles, channels, w, h]
        containing the ensemble predictions.
        :param actuals: Tensor of shape [n_models, n_predictions,
                                         channels, w, h]
        containing the actual values.
        :return: RMSE for each model, prediction, and channel, and total RMSE
         across all channels.
        """
        # Calculating the ensemble mean along the n_ensembles dimension
        ensemble_mean = torch.mean(predictions, dim=2)

        # Calculating RMSE for each channel
        mse_channel = torch.mean((ensemble_mean - actuals) ** 2, dim=[3, 4])
        # Mean over spatial dimensions w and h
        rmse_channel = torch.sqrt(mse_channel)

        # Calculating total RMSE across all channels
        mse_total = torch.mean((ensemble_mean - actuals) ** 2, dim=[2, 3, 4])
        # Mean over channels, w, and h dimensions
        rmse_total = torch.sqrt(mse_total)

        return rmse_channel, rmse_total

    def Spread_skil_ratio(self, predictions, actuals):
        """
        Calculate Spread/Skill ratio for ensemble predictions.

        :param predictions: Tensor of shape [n_models, n_predictions,
                                             n_ensembles, channels, w, h]
        containing the ensemble predictions.
        :param actuals: Tensor of shape [n_models, n_predictions,
                                         channels, w, h]
         containing the actual values.
        :return: Spread/Skill ratio for each channel and total.
        """
        rmse_channel, rmse_total = self.EMRMS(predictions, actuals)

        # Calculating Spread (standard deviation of the ensembles)
        n_ensamble = predictions.shape[2]
        spread_constant = torch.tensor((n_ensamble+1)/n_ensamble)
        if n_ensamble > 1:
            spread_channel = torch.std(predictions, dim=2)
            spread_total = torch.sqrt(torch.mean(spread_channel ** 2,
                                                 dim=[2, 3, 4]))
        else:
            # Standard deviation is zero if there is only one ensemble member
            spread_total = torch.zeros_like(rmse_total)

        # Calculating Spread/Skill ratio
        # Avoiding division by zero by adding a small epsilon where necessary

        epsilon = 1e-6
        spread_skill_ratio_total = torch.sqrt(spread_constant)*spread_total / (rmse_total + epsilon)

        return spread_skill_ratio_total

    def binarize_data(self, data, threshold_dict):
        """ Binarize data based on the given threshold dictionary with
        separate thresholds per channel. """
        binary_data = torch.zeros_like(data)
        # Assume that threshold_dict is a dictionary with keys '<=' and '>='
        # each having a list of thresholds per channel.
        for operator, thresholds in threshold_dict.items():
            for ch, value in enumerate(thresholds):
                if operator == '<=':
                    binary_data[:, :, ch, :, :] = data[:, :, ch, :, :] <= value
                elif operator == '>=':
                    binary_data[:, :, ch, :, :] = data[:, :, ch, :, :] >= value
        return binary_data.float()

    def calculate_and_plot_roc(self, predictions, actuals, threshold_dict):
        """ Calculate ROC for each channel and plot them. """
        # Binarize predictions and actuals
        predictions_binarized = self.binarize_data(predictions.mean(dim=2),
                                                   threshold_dict)
        actuals_binarized = self.binarize_data(actuals, threshold_dict)
        print(actuals_binarized.shape)
        print(predictions_binarized.shape)

        n_models, n_channels = (predictions_binarized.shape[0],
                                predictions_binarized.shape[2])

        # ROC for each model
        for model in range(n_models):
            plt.figure(figsize=(10, 8))
            # Calculate ROC for each channel
            for ch in range(n_channels):
                # Flatten the data to fit roc_curve function requirements
                y_true = actuals_binarized[:, :, ch, :, :].flatten()
                y_score = predictions_binarized[model, :, ch, :, :].flatten()

                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)

                # Plot ROC curve
                plt.plot(fpr, tpr, linestyle='-', label=f'Channel {ch + 1} (AUC = {roc_auc:.2f})')

            plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve of model {model} by Channel')
            plt.legend(loc='lower right')
            plt.show()
