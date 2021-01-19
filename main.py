import feature_eng

if __name__ =="__main__":

    print('Heelo')

    df = pd.read_csv("/Users/aidandempster/projects/uhn/vidProc/demo/preprocessed.csv")
    test_metrics = StrokeMetrics(df, rest_frames=range(10, 100), active_frames=range(300, 600))
    metrics = test_metrics.compute_metrics()
    print(metrics)
