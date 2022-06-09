import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, norm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def sales_per_group_hist( col_name: str, y: str, data , x_label: str=None, y_label: str=None, plot_title: str=None ):
    grouped_data = data.groupby(col_name)[y].sum().reset_index()

    fig, ax = plt.subplots( figsize=(13,7) )
    sns.barplot( x=col_name, y=y, data=grouped_data, color="mediumblue", ax=ax )

    x_label = "Groups" if x_label == None else x_label
    y_label = "Sales" if y_label == None else y_label
    plot_title = "Total Sales" if plot_title == None else plot_title

    ax.set(
        xlabel = x_label,
        ylabel = y_label,
        title = plot_title
    )

    sns.despine( fig=fig, ax=ax )
    return fig


def correlation_plot(data , col_names=None, fontSize=11):
    data = data.T
    ds = data.shape
    col_names = [f"var_{i+1}" for i in range(ds[0])] if col_names == None else col_names 
    
    fig,ax = plt.subplots(nrows=ds[0], ncols=ds[0],figsize=(ds[0],ds[0]))
    
    # Changing the number of ticks per subplot
    for axi in ax.flat:
        axi.xaxis.set_major_locator(plt.MaxNLocator(2))
        axi.yaxis.set_major_locator(plt.MaxNLocator(2))                        
    
    # plotting each subplot             
    for i in range(ds[0]):
        for j in range(ds[0]):
            if i == j:
                # plotting histograms of each variable
                n, bins, patches=ax[i,j].hist(data[i],density=True)
                
                # plotting distribution function and using it to fit a gaussian
                mu, std = norm.fit(data[i])
                p = norm.pdf(bins, mu, std)
                ax[i,j].plot(bins, p, 'r--', linewidth=2)
                ax[i,j].set_xticks([])
                ax[i,j].set_yticks([])
                if j == ds[0]-1:
                    ax[i,j].set_ylabel(col_names[i],fontsize=fontSize).set_color("red")
                    ax[i,j].yaxis.set_label_position("right")
                
                if i == 0 and j == 0:
                    ax[i,j].set_title(col_names[i],fontsize=fontSize).set_color("red")
    
            elif i < j:
                prs=pearsonr(data[i],data[j])[0]
                if prs >= 0.5 or prs <= -0.5:
                    ax[i,j].text(0.5,0.5,str(prs)[0:4],fontsize=24,horizontalalignment='center',verticalalignment='center')                      
                    ax[i,j].text(0.8,0.8,"***",color='r',fontsize=16,horizontalalignment='center',verticalalignment='center')                      
                elif (prs <= -0.45 and prs >= -0.50) or (prs >= 0.45 and prs <= 0.50):
                    ax[i,j].text(0.5,0.5,str(prs)[0:4],fontsize=18,horizontalalignment='center',verticalalignment='center')                      
                    ax[i,j].text(0.8,0.8,"**",color='r',fontsize=16,horizontalalignment='center',verticalalignment='center')                      
                elif (prs <= -0.4 and prs > -0.45) or (prs >= 0.4 and prs < 0.45):
                    ax[i,j].text(0.5,0.5,str(prs)[0:4],fontsize=16,horizontalalignment='center',verticalalignment='center')                      
                    ax[i,j].text(0.8,0.8,"*",color='r',fontsize=16,horizontalalignment='center',verticalalignment='center')
                else:                    
                    ax[i,j].text(0.5,0.5,str(pearsonr(data[i],data[j])[0])[0:4],fontsize=10,horizontalalignment='center',verticalalignment='center')                      
    
                ax[i,j].set_xticks([])
                ax[i,j].set_yticks([])
    
                if i == 0:
                    ax[i,j].set_title(col_names[j],fontsize=fontSize).set_color("red")
                    ax[i,j].set_xticks([])
                    ax[i,j].set_yticks([])
                if j == ds[0]-1:
                    ax[i,j].set_ylabel(col_names[i],fontsize=fontSize).set_color("red")
                    ax[i,j].yaxis.set_label_position("right")
    
            elif i > j:
                ax[i,j].scatter(data[i],data[j],s=10,c='k')      
                rnge= data[i].max()-data[i].min()
                ax[i,j].set_ylim(-0.2*rnge,1.2*rnge)
                ax[i,j].set_xlim(-0.2*rnge,1.2*rnge)                      
                    
                if i!=0 and i!=ds[0]-1:
                    if j==0:
                        ax[i,j].set_xticks([])
                    elif j!=0:
                        ax[i,j].set_xticks([])
                        ax[i,j].set_yticks([])
                        
                if j!=0 and j!=ds[0]-1 and i==ds[0]-1:
                    ax[i,j].set_yticks([])
    
    plt.subplots_adjust(wspace=0, hspace=0)
    return fig


# def show_plots( product, labels_y, dates_x, group_names ):
#     format, subformat, product_id = group_names
#     years = [2021,2021,2020,2020]
#     fig, ax = plt.subplots( len(labels_y), figsize = (12,12) )
#     for i in range(len(labels_y)):
#         sns.lineplot( x = dates_x[i], y = labels_y[i], data=product, ax=ax[i], color="mediumblue", label=f"Total {labels_y[i]}")

#         monthly_mean = product.groupby(product[dates_x[i]].dt.month)[labels_y[i]].mean().reset_index()
#         monthly_mean[dates_x[i]] = monthly_mean[dates_x[i]].apply(lambda x : str(years[i])+"-"+str(x)+"-15")
#         monthly_mean[dates_x[i]] = pd.to_datetime(monthly_mean[dates_x[i]])
#         sns.lineplot( x=dates_x[i], y=labels_y[i], data=monthly_mean, ax=ax[i], color='red', label=f'Mean {labels_y[i]}')
#         ax[i].set(
#             xlabel = f"{dates_x[i]}",
#             ylabel = f"{labels_y[i]}",
#             title = f"{labels_y[i]} {years[i]}"
#         )
    
#     fig.suptitle('Product: {}, Store: {}'.format(product_id, format + " " + subformat))
#     fig.tight_layout(rect=[0, 0, 1, 0.94])
#     return fig

def show_plots( product, labels_y, group_names = (0, 0), year = 2021 ):
    product = product.copy()
    product = product.reset_index()
    product["py_dates"] = product["dates"] + pd.offsets.DateOffset(years=-1)
    id_store, id_product  = group_names
    fig, axs = plt.subplots( len(labels_y), figsize = (12,12) )

    for label,ax in zip(labels_y,axs):

        date = "py_dates" if label[:3] == "py_" else "dates"
        Y = year-1 if label[:3] == "py_" else year

        sns.lineplot( x = date, y = label, data=product, ax=ax, color="mediumblue", label=f"Total {label}")

        monthly_mean = product.groupby(product[date].dt.month)[label].mean().reset_index()
        monthly_mean[date] = monthly_mean[date].apply(lambda x : str(Y)+"-"+str(x)+"-15")
        monthly_mean[date] = pd.to_datetime(monthly_mean[date])

        sns.lineplot( x=date, y=label, data=monthly_mean, ax=ax, color='red', label=f'Mean {label}')

        ax.set(
            xlabel = f"{date}",
            ylabel = f"{label}",
            title = f"{label} {Y}"
        )
    
    fig.suptitle('Product: {}, Store: {}'.format(id_product, id_store))
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def ac_pac_plot( data, label, lags=None ):
    dt_data = data[label]

    fig, ax = plt.subplots(ncols=3, figsize=(12,4))

    dt_data.plot( ax=ax[0], color="mediumblue")
    plot_acf( dt_data, lags=lags, ax=ax[1], color="mediumblue")
    plot_pacf( dt_data, lags=lags, ax=ax[2], color="mediumblue", method="ywm")

    sns.despine()
    plt.tight_layout()
    return fig


def plot_aggr_data( df, feature ):
    df = df.copy()
    df = df.reset_index()

    fig, ax = plt.subplots( figsize=(12, 3) )

    df.groupby("dates").mean()[feature].plot( ax=ax, title=f"{feature} TS (Aggregated data)", legend=True)
    df.groupby("dates").mean()[f"py_{feature}"].plot(ax=ax, legend=True)

    plt.tight_layout()

    return fig


def plot_seasonality_patterns( df_dates_categorized, feature ):
    df = df_dates_categorized.copy()
    df = df.reset_index()
    
    fig, axs = plt.subplots( ncols=2, nrows=2, figsize=(10,10) )
    _ = pd.pivot_table( df, values=feature, columns='Year', index='Month').plot(title="Yearly seasonality", ax=axs[0,0])
    _ = pd.pivot_table( df, values=f"py_{feature}", columns='Year', index='Month').plot(title="Yearly seasonality", ax=axs[0,0])

    _ = pd.pivot_table( df, values=feature, columns='Month', index='Day').plot(title="Monthly seasonality", ax=axs[0,1])

    _ = pd.pivot_table( df, values=feature, columns='Year', index='Dayofweek').plot(title="Weekly seasonality (by year)", ax=axs[1,0])
    _ = pd.pivot_table( df, values=f"py_{feature}", columns='Year', index='Dayofweek').plot(title="Weekly seasonality (by year)", ax=axs[1,0])

    _ = pd.pivot_table( df, values=feature, columns='Month', index='Dayofweek').plot(title="Weekly seasonality (by month)", ax=axs[1,1])

    fig.suptitle(f'{feature} seasonality patterns (aggregated data)')
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


def plot_volatility( data, feature ):
    df = data.copy()
    # df = df.reset_index()

    df[feature] /= data.groupby(["id_products", "id_stores"])[feature].transform("mean")

    fig, axs = plt.subplots( 2, figsize=(12,8) )
    _ = df.groupby("dates")[feature].std().plot( ax=axs[0], title='Volatility (across products and stores)' )
    _ = (df.groupby(['id_stores', 'id_products'])[['dates', feature]].rolling(7, on='dates').std().groupby('dates').mean().plot( ax=axs[1], title='Volatility (7-d rolling, aggregated data)'))

    return fig


def plot_trend( df, features ):
    fig, ax = plt.subplots( figsize=(5,5))
    for feature in features:
        _ = pd.pivot_table( df, values=feature, index='Year').plot(ax=ax, style='-o', title="Annual trend (aggregated data)")
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("success")