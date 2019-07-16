import argparse

from EpiDeepWrapper import epideep

def main():

    parser = argparse.ArgumentParser(description="EpiDeep")
    
    parser.add_argument('--start_week',type=int, default='20',help='The beginning week of the prediction')
    parser.add_argument('--end_week',type=int, default='21',help='The last week of the prediction')
    parser.add_argument('--start_year', type=int, default='2000',help='The first season/year in the used data')
    parser.add_argument('--end_year', type=int, default='2017', help='The current season/year')
    parser.add_argument('--pred_metrics', type=str, default='Future', help="The epidemiological metrics of the prediction, i.e., Future, Peak, Peak_time, and Onset")
    parser.add_argument('--eval_metrics', type=str, default='RMSE', help="The evaluation metrics, i.e., RMSE, MAPE, and logscore")
    parser.add_argument('--region', type=str, default='National', help="The region for the epidemic prediction")
    parser.add_argument('--iterations',type=int, default='1000',help='No. of training iterations')
    args = parser.parse_args()

    epideep(args)

    


if __name__=='__main__':
    main()
