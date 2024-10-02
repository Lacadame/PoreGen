class Ocean_Plot():
    def __init__(self):
        pass

    def plot_graphs(data,predictions_list,ind_array,D,skip,latitudes,longitudes, subplot = [3,3]):
        nlin, ncol = subplot

        # Vector Field

        # Parameters for adjusting quiver plots
        quiver_scale = 30  # Adjust this value as needed for scaling arrow size
        quiver_alpha = 0.8  # Adjust this value as needed for arrow transparency
        # plot predictions vs data
        fig, ax = plt.subplots(nlin, ncol, figsize=(ncol*3, nlin*3))
        for i, ind in enumerate(ind_array):
            mask_values = mask.cpu().numpy()
            masked_u = np.where(mask_values == 0, data[ind][0].cpu().numpy(), np.nan)
            masked_v = np.where(mask_values == 0, data[ind][1].cpu().numpy(), np.nan)

            ind2 = ind-D-1

            # Overlay vector field using quiver
            ax[0, i].quiver(longitudes.cpu().numpy(), latitudes.cpu().numpy(),
                            masked_u, masked_v,
                            scale=quiver_scale, alpha=quiver_alpha, color='black')
            ax[0, i].set_title(f"h={hour_since_year_start+ind*skip} (True)")
            # ax[0, i].axis('off')

            for j in range(nlin -1):
                
                masked_u = np.where(mask_values == 0,
                                    predictions_list[j][ind2][0].cpu().numpy(), np.nan)
                masked_v = np.where(mask_values == 0,
                                    predictions_list[j][ind2][1].cpu().numpy(), np.nan)
        
                ax[j+1, i].quiver(longitudes.cpu().numpy(), latitudes.cpu().numpy(),
                                masked_u,
                                masked_v,
                                scale=quiver_scale, alpha=quiver_alpha, color='black')
                
                
                ax[j+1, i].set_title(f"h={hour_since_year_start+ind*skip} (Gen {j+1})")
                # ax[j+1, i].axis('off')
        plt.suptitle(f'Vector Field (U and V) for D={D} and skip={skip}')

        # Set latitude and longitude labels with degrees
        for axs in ax[:, 0]:  # Only the first column gets y-axis labels
            axs.set_ylabel('Latitude (°)')
        for axs in ax[-1, :]:  # Only the bottom row gets x-axis labels
            axs.set_xlabel('Longitude (°)')

        fig.tight_layout()
        plt.show()

        # # ##################################### Vector + SSH

        fig, ax = plt.subplots(nlin, ncol, figsize=(ncol*3, nlin*3))


        vmin = min([np.min(data[i][2].cpu().numpy()) for i in ind_array])
        vmax = max([np.max(data[i][2].cpu().numpy()) for i in ind_array])

        for i, ind in enumerate(ind_array):
            ind2 = ind-D-1
            data_values = data[ind][2].cpu().numpy()
            mask_values = mask.cpu().numpy()
            masked_data = np.where(mask_values == 0, data_values, np.nan)

            masked_u = np.where(mask_values == 0,
                                data[ind][0].cpu().numpy(), np.nan)
            masked_v = np.where(mask_values == 0,
                                data[ind][1].cpu().numpy(), np.nan)

            # Display scalar field using pcolormesh
            pcm = ax[0, i].pcolormesh(longitudes.cpu().numpy(),
                                    latitudes.cpu().numpy(), masked_data, shading='auto',
                                    vmin = vmin, vmax=vmax)
            # Overlay vector field using quiver
            ax[0, i].quiver(longitudes.cpu().numpy(), latitudes.cpu().numpy(),
                            masked_u, masked_v,
                            scale=quiver_scale, alpha=quiver_alpha, color='black')
            ax[0, i].set_title(f"h={hour_since_year_start+ind*skip} (True)")
            # ax[0, i].axis('off')

            for j in range(nlin -1):

                masked_pred = np.where(mask_values == 0,
                                    predictions_list[j][ind2][2].cpu().numpy(), np.nan)
                masked_u = np.where(mask_values == 0,
                                    predictions_list[j][ind2][0].cpu().numpy(), np.nan)
                masked_v = np.where(mask_values == 0,
                                    predictions_list[j][ind2][1].cpu().numpy(), np.nan)

                ax[j+1, i].pcolormesh(longitudes.cpu().numpy(),
                                    latitudes.cpu().numpy(),
                                    masked_pred, shading='auto')
                ax[j+1, i].quiver(longitudes.cpu().numpy(), latitudes.cpu().numpy(),
                                masked_u,
                                masked_v,
                                scale=quiver_scale, alpha=quiver_alpha, color='black')
                ax[j+1, i].set_title(f"h={hour_since_year_start+ind*skip} (Gen {j+1})")
                # ax[j+1, i].axis('off')

        # Add colorbar at the bottom of the figure
        cbar_ax = fig.add_axes([0.15, -0.02, 0.7, 0.02])  # Adjust these values as needed
        cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(r'm')

        plt.suptitle(f'Vector Field (U and V) and SSH for D={D} and skip={skip}')


        # Set latitude and longitude labels with degrees
        for axs in ax[:, 0]:  # Only the first column gets y-axis labels
            axs.set_ylabel('Latitude (°)')
        for axs in ax[-1, :]:  # Only the bottom row gets x-axis labels
            axs.set_xlabel('Longitude (°)')

        fig.tight_layout()
        plt.show()


        # ##################################### SSH

            # Create figure and subplots

        fig, ax = plt.subplots(nlin, ncol, figsize=(ncol*3, nlin*3))

        # Determine the common color scale across all plots

        vmin = min([np.min(data[i][2].cpu().numpy()) for i in ind_array])
        vmax = max([np.max(data[i][2].cpu().numpy()) for i in ind_array])

        # plot predictions vs data
        for i, ind in enumerate(ind_array):
            ind2 = ind-D-1

            data_values = data[ind][2].cpu().numpy()
            mask_values = mask.cpu().numpy()
            masked_data = np.where(mask_values == 0, data_values, np.nan)
            
            # Display scalar field using pcolormesh with consistent color scale
            pcm = ax[0, i].pcolormesh(longitudes.cpu().numpy(), latitudes.cpu().numpy(), masked_data,
                                    shading='auto', vmin=vmin, vmax=vmax)
            ax[0, i].set_title(f"h={hour_since_year_start+ind*skip} (True)")
            for j in range(nlin-1):
                masked_data = np.where(mask_values == 0,
                                    predictions_list[j][ind2][2].cpu().numpy(),
                                    np.nan)

                ax[j+1, i].pcolormesh(longitudes.cpu().numpy(), latitudes.cpu().numpy(),
                                    masked_data, shading='auto', vmin=vmin, vmax=vmax)
                ax[j+1, i].set_title(f"h={hour_since_year_start+ind*skip} (Gen {j+1})")

        # Add colorbar at the bottom of the figure
        cbar_ax = fig.add_axes([0.15, -0.02, 0.7, 0.02])  # Adjust these values as needed
        cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(r'm')

        # Set latitude and longitude labels with degrees
        for axs in ax[:, 0]:  # Only the first column gets y-axis labels
            axs.set_ylabel('Latitude (°)')
        for axs in ax[-1, :]:  # Only the bottom row gets x-axis labels
            axs.set_xlabel('Longitude (°)')

        fig.suptitle(f'Sea Surface Height (SSH) for D={D} and Skip={skip}')
        fig.tight_layout()
        plt.show()

        # #################################### U

        ## Create figure and subplots
        fig, ax = plt.subplots(nlin, ncol, figsize=(ncol*3, nlin*3))

        # Determine the common color scale across all plots
        vmin = min([np.min(data[i][0].cpu().numpy()) for i in ind_array])
        vmax = max([np.max(data[i][0].cpu().numpy()) for i in ind_array])

        # plot predictions vs data
        for i, ind in enumerate(ind_array):
            ind2 = ind-D-1

            data_values = data[ind][0].cpu().numpy()
            mask_values = mask.cpu().numpy()
            masked_data = np.where(mask_values == 0, data_values, np.nan)
            
            # Display scalar field using pcolormesh with consistent color scale
            pcm = ax[0, i].pcolormesh(longitudes.cpu().numpy(), latitudes.cpu().numpy(), masked_data,
                                    shading='auto', vmin=vmin, vmax=vmax)
            ax[0, i].set_title(f"h={hour_since_year_start+ind*skip} (True)")
            for j in range(nlin -1):
                masked_data = np.where(mask_values == 0,
                                    predictions_list[j][ind2][0].cpu().numpy(),
                                    np.nan)

                ax[j+1, i].pcolormesh(longitudes.cpu().numpy(), latitudes.cpu().numpy(),
                                    masked_data, shading='auto', vmin=vmin, vmax=vmax)
                ax[j+1, i].set_title(f"h={hour_since_year_start+ind*skip} (Gen {j+1})")

        # Add colorbar at the bottom of the figure
        cbar_ax = fig.add_axes([0.15, -0.02, 0.7, 0.02])  # Adjust these values as needed
        cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(r'$v (m/s)$')

        # Set latitude and longitude labels with degrees
        for axs in ax[:, 0]:  # Only the first column gets y-axis labels
            axs.set_ylabel('Latitude (°)')
        for axs in ax[-1, :]:  # Only the bottom row gets x-axis labels
            axs.set_xlabel('Longitude (°)')

        fig.suptitle(f'Horizontal velocity (U) for D={D} and Skip={skip}')
        fig.tight_layout()
        plt.show()


    # #################################### V

        ## Create figure and subplots
        fig, ax = plt.subplots(nlin, ncol, figsize=(ncol*3, nlin*3))

        # Determine the common color scale across all plots
        vmin = min([np.min(data[i][1].cpu().numpy()) for i in ind_array])
        vmax = max([np.max(data[i][1].cpu().numpy()) for i in ind_array])

        # plot predictions vs data
        for i, ind in enumerate(ind_array):
            ind2 = ind-D-1

            data_values = data[ind][1].cpu().numpy()
            mask_values = mask.cpu().numpy()
            masked_data = np.where(mask_values == 0, data_values, np.nan)
            
            # Display scalar field using pcolormesh with consistent color scale
            pcm = ax[0, i].pcolormesh(longitudes.cpu().numpy(), latitudes.cpu().numpy(), masked_data,
                                    shading='auto', vmin=vmin, vmax=vmax)
            ax[0, i].set_title(f"h={hour_since_year_start+ind*skip} (True)")
            for j in range(nlin -1):
                masked_data = np.where(mask_values == 0,
                                    predictions_list[j][ind2][1].cpu().numpy(),
                                    np.nan)

                ax[j+1, i].pcolormesh(longitudes.cpu().numpy(), latitudes.cpu().numpy(),
                                    masked_data, shading='auto', vmin=vmin, vmax=vmax)
                ax[j+1, i].set_title(f"h={hour_since_year_start+ind*skip} (Gen {j+1})")

        # Add colorbar at the bottom of the figure
        cbar_ax = fig.add_axes([0.15, -0.02, 0.7, 0.02])  # Adjust these values as needed
        cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(r'$v (m/s)$')

        # Set latitude and longitude labels with degrees
        for axs in ax[:, 0]:  # Only the first column gets y-axis labels
            axs.set_ylabel('Latitude (°)')
        for axs in ax[-1, :]:  # Only the bottom row gets x-axis labels
            axs.set_xlabel('Longitude (°)')

        fig.suptitle(f'Vertical velocity (V) for D={D} and Skip={skip}')
        fig.tight_layout()
        plt.show()

    def plot_graphs_tresh(data,predictions_list,ind_array,D,skip,latitudes,longitudes, subplot = [3,3]):
        nlin, ncol = subplot

        # ############################################# SSH 


        # Create figure and subplots
        fig, ax = plt.subplots(nlin, ncol, figsize=(ncol*3, nlin*3))

        # Determine the common color scale across all plots
        vmin = min([np.min(data[i][2].cpu().numpy()) for i in ind_array])
        vmax = max([np.max(data[i][2].cpu().numpy()) for i in ind_array])

        # plot predictions vs data
        thresholds1 = [0,0.4, 1]
        thresholds2 = [-1,0]
        colors1 = ['green','yellow', 'red']
        colors2 = [(73/255,11/255,96/255,225/255), 'blue']

        colors = [(73/255,11/255,96/255,225/255), 'blue', 'green' ,'yellow', 'red']
        cmap1 = ListedColormap(colors)
        norm = BoundaryNorm( [-2, -1, 0, 0.4, 1,2], cmap1.N, clip=True)



        for i, ind in enumerate(ind_array):
            ind2 = ind-D-1
            data_values = data[ind][2].cpu().numpy()
            mask_values = mask.cpu().numpy()
            masked_data = np.where(mask_values == 0, data_values, np.nan)

            ax[0, i].pcolormesh(longitudes.cpu().numpy(), latitudes.cpu().numpy(), masked_data,
                                    shading='auto', vmin = vmin, vmax=vmax)

            pcm = ax[0, i].pcolormesh(longitudes.cpu().numpy(), latitudes.cpu().numpy(), masked_data,
                                    shading='auto', norm= norm, cmap=cmap1)

            for threshold, color in zip(thresholds1, colors1):
                mask_t = np.abs(masked_data) >= threshold
                mask_t = np.where(mask_t, 1, np.nan)
                cmap = ListedColormap([color,'none'])
                ax[0, i].pcolormesh(longitudes.cpu().numpy(), latitudes.cpu().numpy(), mask_t,
                                    cmap=cmap, shading='auto', alpha=1)
                
            for threshold, color in zip(thresholds2, colors2):
                mask_t = np.abs(masked_data) <= threshold
                mask_t = np.where(mask_t, 1, np.nan)
                cmap = ListedColormap([color,'none'])
                ax[0, i].pcolormesh(longitudes.cpu().numpy(), latitudes.cpu().numpy(), mask_t,
                                    cmap=cmap, shading='auto', alpha=1)
            
            ax[0, i].set_title(f"h={hour_since_year_start+ind*skip} (True)")
            for j in range(nlin-1):
                masked_data = np.where(mask_values == 0,
                                    predictions_list[j][ind2][2].cpu().numpy(),
                                    np.nan)

                ax[j+1, i].pcolormesh(longitudes.cpu().numpy(), latitudes.cpu().numpy(),
                                    masked_data, shading='auto', vmin=vmin, vmax=vmax)
                
                for threshold, color in zip(thresholds1, colors1):
                    mask_t = np.abs(masked_data) >= threshold
                    mask_t = np.where(mask_t, 1, np.nan)
                    cmap = ListedColormap([color,'none'])
                    ax[j+1, i].pcolormesh(longitudes.cpu().numpy(), latitudes.cpu().numpy(), mask_t,
                                        cmap=cmap, shading='auto', alpha=1)
                    
                for threshold, color in zip(thresholds2, colors2):
                    mask_t = np.abs(masked_data) <= threshold
                    mask_t = np.where(mask_t, 1, np.nan)
                    cmap = ListedColormap([color,'none'])
                    ax[0, i].pcolormesh(longitudes.cpu().numpy(), latitudes.cpu().numpy(), mask_t,
                                        cmap=cmap, shading='auto', alpha=1)
                
                ax[j+1, i].set_title(f"h={hour_since_year_start+ind*skip} (Gen {j+1})")

        
        cbar_ax = fig.add_axes([0.15, -0.02, 0.7, 0.02])  # Adjust these values as needed
        cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal', ticks = [-1.5, -0.5, 0.2 ,0.7, 1.5] )
        cbar.set_ticklabels(['$\\leq-1$', '$-1\\leq 0$', '$0\\leq0.4$' ,'$0.4\\leq1$', '$\\leq1$'])
        cbar.set_label(r'm')

        # Set latitude and longitude labels with degrees
        for axs in ax[:, 0]:  # Only the first column gets y-axis labels
            axs.set_ylabel('Latitude (°)')
        for axs in ax[-1, :]:  # Only the bottom row gets x-axis labels
            axs.set_xlabel('Longitude (°)')

        fig.suptitle(f'Sea Surface Height (SSH) with thresholds for D={D} and Skip={skip}')
        fig.tight_layout()
        plt.show()

        # ############################################## U


        ## Create figure and subplots
        fig, ax = plt.subplots(nlin, ncol, figsize=(ncol*3, nlin*3))

        # Determine the common color scale across all plots
        vmin = min([np.min(data[i][0].cpu().numpy()) for i in ind_array])
        vmax = max([np.max(data[i][0].cpu().numpy()) for i in ind_array])
        # plot predictions vs data
        # Define thresholds and corresponding colors
        thresholds = [0, 0.1, 0.3, 0.5, 1]  # Adding np.inf to cover all possible data points above the last threshold
        colors = [(73/255,11/255,96/255,225/255), 'green', 'yellow', 'red']
        cmap1 = ListedColormap(colors)
        norm = BoundaryNorm(thresholds, cmap1.N, clip=True)


        for i, ind in enumerate(ind_array):
            ind2 = ind-D-1
            data_values = data[ind][0].cpu().numpy()
            mask_values = mask.cpu().numpy()
            masked_data = np.where(mask_values == 0, data_values, np.nan)

            pcm = ax[0, i].pcolormesh(longitudes.cpu().numpy(), latitudes.cpu().numpy(), masked_data,
                                    shading='auto', norm = norm, cmap=cmap1)

            for threshold, color in zip(thresholds, colors):
                mask_t = np.abs(masked_data) >= threshold
                mask_t = np.where(mask_t, 1, np.nan)
                cmap = ListedColormap([color,'none'])
                ax[0, i].pcolormesh(longitudes.cpu().numpy(), latitudes.cpu().numpy(), mask_t,
                                    cmap=cmap, shading='auto', alpha=1)

            
            ax[0, i].set_title(f"h={hour_since_year_start+ind*skip} (True)")
            for j in range(nlin -1):
                velocity_vec = predictions_list[j][ind2][0].cpu().numpy()
                masked_data = np.where(mask_values == 0,
                                    velocity_vec,
                                    np.nan)

                ax[j+1, i].pcolormesh(longitudes.cpu().numpy(), latitudes.cpu().numpy(),
                                    masked_data, shading='auto', vmin=vmin, vmax=vmax)
                
                for threshold, color in zip(thresholds, colors):
                    mask_t = np.abs(masked_data) >= threshold
                    mask_t = np.where(mask_t, 1, np.nan)
                    cmap = ListedColormap([color,'none'])
                    ax[j+1, i].pcolormesh(longitudes.cpu().numpy(), latitudes.cpu().numpy(), mask_t,
                                        cmap=cmap, shading='auto', alpha=1)
                
                ax[j+1, i].set_title(f"h={hour_since_year_start+ind*skip} (Gen {j+1})")

        cbar_ax = fig.add_axes([0.15, -0.02, 0.7, 0.02])  # Adjust these values as needed
        cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal', ticks=[0.05, 0.2, 0.4, 0.75])
        cbar.set_ticklabels(['$0\\leq 0.1$', '$0.1\\leq 0.3$', '$0.3\\leq 0.5$', '$0.5\\leq$'])
        cbar.set_label(r'm/s')


        # Set latitude and longitude labels with degrees
        for axs in ax[:, 0]:  # Only the first column gets y-axis labels
            axs.set_ylabel('Latitude (°)')
        for axs in ax[-1, :]:  # Only the bottom row gets x-axis labels
            axs.set_xlabel('Longitude (°)')

        fig.suptitle(f'Horizontal Velocity (U) with thresholds for D={D} and Skip={skip}')
        fig.tight_layout()
        plt.show()


        # ############################################## V

        # Create figure and subplots
        fig, ax = plt.subplots(nlin, ncol, figsize=(ncol*3, nlin*3))

        # Determine the common color scale across all plots
        vmin = min([np.min(data[i][1].cpu().numpy()) for i in ind_array])
        vmax = max([np.max(data[i][1].cpu().numpy()) for i in ind_array])
        # plot predictions vs data
        # Define thresholds and corresponding colors
        thresholds = [0, 0.1, 0.3, 0.5, 1]  # Adding np.inf to cover all possible data points above the last threshold
        colors = [(73/255,11/255,96/255,225/255), 'green', 'yellow', 'red']
        cmap1 = ListedColormap(colors)
        norm = BoundaryNorm(thresholds, cmap1.N, clip=True)


        for i, ind in enumerate(ind_array):
            ind2 = ind-D-1
            data_values = data[ind][1].cpu().numpy()
            mask_values = mask.cpu().numpy()
            masked_data = np.where(mask_values == 0, data_values, np.nan)

            pcm = ax[0, i].pcolormesh(longitudes.cpu().numpy(), latitudes.cpu().numpy(), masked_data,
                                    shading='auto', norm = norm, cmap=cmap1)

            for threshold, color in zip(thresholds, colors):
                mask_t = np.abs(masked_data) >= threshold
                mask_t = np.where(mask_t, 1, np.nan)
                cmap = ListedColormap([color,'none'])
                ax[0, i].pcolormesh(longitudes.cpu().numpy(), latitudes.cpu().numpy(), mask_t,
                                    cmap=cmap, shading='auto', alpha=1)

            
            ax[0, i].set_title(f"h={hour_since_year_start+ind*skip} (True)")
            for j in range(nlin -1):
                velocity_vec = predictions_list[j][ind2][1].cpu().numpy()
                masked_data = np.where(mask_values == 0,
                                    velocity_vec,
                                    np.nan)

                ax[j+1, i].pcolormesh(longitudes.cpu().numpy(), latitudes.cpu().numpy(),
                                    masked_data, shading='auto', vmin=vmin, vmax=vmax)
                
                for threshold, color in zip(thresholds, colors):
                    mask_t = np.abs(masked_data) >= threshold
                    mask_t = np.where(mask_t, 1, np.nan)
                    cmap = ListedColormap([color,'none'])
                    ax[j+1, i].pcolormesh(longitudes.cpu().numpy(), latitudes.cpu().numpy(), mask_t,
                                        cmap=cmap, shading='auto', alpha=1)
                
                ax[j+1, i].set_title(f"h={hour_since_year_start+ind*skip} (Gen {j+1})")

        cbar_ax = fig.add_axes([0.15, -0.02, 0.7, 0.02])  # Adjust these values as needed
        cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal', ticks=[0.05, 0.2, 0.4, 0.75])
        cbar.set_ticklabels(['$0\\leq 0.1$', '$0.1\\leq 0.3$', '$0.3\\leq 0.5$', '$0.5\\leq$'])
        cbar.set_label(r'm/s')


        # Set latitude and longitude labels with degrees
        for axs in ax[:, 0]:  # Only the first column gets y-axis labels
            axs.set_ylabel('Latitude (°)')
        for axs in ax[-1, :]:  # Only the bottom row gets x-axis labels
            axs.set_xlabel('Longitude (°)')

        fig.suptitle(f'Vertical Velocity (V) with thresholds for D={D} and Skip={skip}')
        fig.tight_layout()
        plt.show()


        # ################## U+V

        # Create figure and subplots
        fig, ax = plt.subplots(nlin, ncol, figsize=(ncol*3, nlin*3))

        # Determine the common color scale across all plots
        vmin = min([np.min(data[i][2].cpu().numpy()) for i in ind_array])
        vmax = max([np.max(data[i][2].cpu().numpy()) for i in ind_array])
        # plot predictions vs data
        # Define thresholds and corresponding colors
        thresholds = [0, 0.1, 0.3, 0.5, 1]  # Adding np.inf to cover all possible data points above the last threshold
        colors = [(73/255,11/255,96/255,225/255), 'green', 'yellow', 'red']
        cmap1 = ListedColormap(colors)
        norm = BoundaryNorm(thresholds, cmap1.N, clip=True)


        for i, ind in enumerate(ind_array):
            ind2 = ind-D-1
            data_values = np.sqrt(data[ind][1].cpu().numpy()**2 + data[ind][1].cpu().numpy()**2)
            mask_values = mask.cpu().numpy()
            masked_data = np.where(mask_values == 0, data_values, np.nan)

            pcm = ax[0, i].pcolormesh(longitudes.cpu().numpy(), latitudes.cpu().numpy(), masked_data,
                                    shading='auto', norm = norm, cmap=cmap1)

            for threshold, color in zip(thresholds, colors):
                mask_t = np.abs(masked_data) >= threshold
                mask_t = np.where(mask_t, 1, np.nan)
                cmap = ListedColormap([color,'none'])
                ax[0, i].pcolormesh(longitudes.cpu().numpy(), latitudes.cpu().numpy(), mask_t,
                                    cmap=cmap, shading='auto', alpha=1)

            
            ax[0, i].set_title(f"h={hour_since_year_start+ind*skip} (True)")
            for j in range(nlin -1):
                velocity_vec = np.sqrt(predictions_list[j][ind2][1].cpu().numpy()**2 + predictions_list[j][ind2][0].cpu().numpy()**2)
                masked_data = np.where(mask_values == 0,
                                    velocity_vec,
                                    np.nan)

                ax[j+1, i].pcolormesh(longitudes.cpu().numpy(), latitudes.cpu().numpy(),
                                    masked_data, shading='auto', vmin=vmin, vmax=vmax)
                
                for threshold, color in zip(thresholds, colors):
                    mask_t = np.abs(masked_data) >= threshold
                    mask_t = np.where(mask_t, 1, np.nan)
                    cmap = ListedColormap([color,'none'])
                    ax[j+1, i].pcolormesh(longitudes.cpu().numpy(), latitudes.cpu().numpy(), mask_t,
                                        cmap=cmap, shading='auto', alpha=1)
                
                ax[j+1, i].set_title(f"h={hour_since_year_start+ind*skip} (Gen {j+1})")

        cbar_ax = fig.add_axes([0.15, -0.02, 0.7, 0.02])  # Adjust these values as needed
        cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal', ticks=[0.05, 0.2, 0.4, 0.75])
        cbar.set_ticklabels(['$0\\leq 0.1$', '$0.1\\leq 0.3$', '$0.3\\leq 0.5$', '$0.5\\leq$'])
        cbar.set_label(r'm/s')


        # Set latitude and longitude labels with degrees
        for axs in ax[:, 0]:  # Only the first column gets y-axis labels
            axs.set_ylabel('Latitude (°)')
        for axs in ax[-1, :]:  # Only the bottom row gets x-axis labels
            axs.set_xlabel('Longitude (°)')

        fig.suptitle(f'Velocity with with thresholds for D={D} and Skip={skip}')
        fig.tight_layout()
        plt.show()

    def plot_graphs_relative(data,predictions_list,ind_array,D,skip,latitudes,longitudes, subplot = [3,3]):
        nlin, ncol = subplot

        # ###################################### SSH
        
        # Create figure and subplots
        fig, ax = plt.subplots(nlin, ncol, figsize=(ncol*3, nlin*3))

        for i, ind in enumerate(ind_array):
            ind2 = ind-D-1

            data_values = data[ind][2].cpu().numpy()
            mask_values = mask.cpu().numpy()
            masked_data = np.where(mask_values == 0, data_values, np.nan)
            
            for j in range(nlin):
                masked_data_err = np.abs(np.where(mask_values == 0,
                                    predictions_list[j][ind2][2].cpu().numpy(),
                                    np.nan) - masked_data)/masked_data
                

                pcm = ax[j, i].pcolormesh(longitudes.cpu().numpy(), latitudes.cpu().numpy(),
                                    masked_data_err, shading='auto', vmin=-1, vmax=1)
                ax[j, i].set_title(f"h={hour_since_year_start+ind*skip} (Gen {j+1})")

        # Add colorbar at the bottom of the figure
        cbar_ax = fig.add_axes([0.15, -0.02, 0.7, 0.02])  # Adjust these values as needed
        cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(r'%')
        
        # Set latitude and longitude labels with degrees
        for axs in ax[:, 0]:  # Only the first column gets y-axis labels
            axs.set_ylabel('Latitude (°)')
        for axs in ax[-1, :]:  # Only the bottom row gets x-axis labels
            axs.set_xlabel('Longitude (°)')

        fig.suptitle(f'Relative Error of Sea Surface Height (SSH) for D={D} and Skip={skip}')
        fig.tight_layout()
        plt.show()

        # ###################################### U


        # Create figure and subplots
        fig, ax = plt.subplots(nlin, ncol, figsize=(ncol*3, nlin*3))

        for i, ind in enumerate(ind_array):
            ind2 = ind-D-1

            data_values = data[ind][0].cpu().numpy()
            mask_values = mask.cpu().numpy()
            masked_data = np.where(mask_values == 0, data_values, np.nan)
            
            for j in range(nlin):
                masked_data_err = np.abs(np.where(mask_values == 0,
                                    predictions_list[j][ind2][0].cpu().numpy(),
                                    np.nan) - masked_data)/masked_data
                

                pcm = ax[j, i].pcolormesh(longitudes.cpu().numpy(), latitudes.cpu().numpy(),
                                    masked_data_err, shading='auto', vmin=-1, vmax=1)
                ax[j, i].set_title(f"h={hour_since_year_start+ind*skip} (Gen {j+1})")

        # Add colorbar at the bottom of the figure
        cbar_ax = fig.add_axes([0.15, -0.02, 0.7, 0.02])  # Adjust these values as needed
        cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(r'%')
        
        # Set latitude and longitude labels with degrees
        for axs in ax[:, 0]:  # Only the first column gets y-axis labels
            axs.set_ylabel('Latitude (°)')
        for axs in ax[-1, :]:  # Only the bottom row gets x-axis labels
            axs.set_xlabel('Longitude (°)')

        fig.suptitle(f'Relative Error of Horizontal Velocity (U) for D={D} and Skip={skip}')
        fig.tight_layout()
        plt.show()

        # ###################################### V


        # Create figure and subplots
        fig, ax = plt.subplots(nlin, ncol, figsize=(ncol*3, nlin*3))

        for i, ind in enumerate(ind_array):
            ind2 = ind-D-1

            data_values = data[ind][1].cpu().numpy()
            mask_values = mask.cpu().numpy()
            masked_data = np.where(mask_values == 0, data_values, np.nan)
            
            for j in range(nlin):
                masked_data_err = np.abs(np.where(mask_values == 0,
                                    predictions_list[j][ind2][1].cpu().numpy(),
                                    np.nan) - masked_data)/masked_data
                

                pcm = ax[j, i].pcolormesh(longitudes.cpu().numpy(), latitudes.cpu().numpy(),
                                    masked_data_err, shading='auto', vmin=-1, vmax=1)
                ax[j, i].set_title(f"h={hour_since_year_start+ind*skip} (Gen {j+1})")

        # Add colorbar at the bottom of the figure
        cbar_ax = fig.add_axes([0.15, -0.02, 0.7, 0.02])  # Adjust these values as needed
        cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(r'%')
        
        # Set latitude and longitude labels with degrees
        for axs in ax[:, 0]:  # Only the first column gets y-axis labels
            axs.set_ylabel('Latitude (°)')
        for axs in ax[-1, :]:  # Only the bottom row gets x-axis labels
            axs.set_xlabel('Longitude (°)')

        fig.suptitle(f'Relative Error of Vertical Velocity (V) for D={D} and Skip={skip}')
        fig.tight_layout()
        plt.show()
            

        # ###################################### U+V


        # Create figure and subplots
        fig, ax = plt.subplots(nlin, ncol, figsize=(ncol*3, nlin*3))

        for i, ind in enumerate(ind_array):
            ind2 = ind-D-1

            data_values = np.sqrt(data[ind][1].cpu().numpy()**2 +  data[ind][0].cpu().numpy()**2)
            mask_values = mask.cpu().numpy()
            masked_data = np.where(mask_values == 0, data_values, np.nan)
            
            for j in range(nlin):
                pred_vel = np.sqrt(predictions_list[j][ind2][1].cpu().numpy()**2 + predictions_list[j][ind2][0].cpu().numpy()**2)
                masked_data_err = np.abs(np.where(mask_values == 0,
                                    pred_vel,
                                    np.nan) - masked_data)/masked_data
                

                pcm = ax[j, i].pcolormesh(longitudes.cpu().numpy(), latitudes.cpu().numpy(),
                                    masked_data_err, shading='auto', vmin=-1, vmax=1)
                ax[j, i].set_title(f"h={hour_since_year_start+ind*skip} (Gen {j+1})")

        # Add colorbar at the bottom of the figure
        cbar_ax = fig.add_axes([0.15, -0.02, 0.7, 0.02])  # Adjust these values as needed
        cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(r'%')
        
        # Set latitude and longitude labels with degrees
        for axs in ax[:, 0]:  # Only the first column gets y-axis labels
            axs.set_ylabel('Latitude (°)')
        for axs in ax[-1, :]:  # Only the bottom row gets x-axis labels
            axs.set_xlabel('Longitude (°)')

        fig.suptitle(f'Relative Error Velocity for D={D} and Skip={skip}')
        fig.tight_layout()
        plt.show()
            
            
        
        


        