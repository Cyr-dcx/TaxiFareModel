# imports
from TaxiFareModel.data import get_data, clean_data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        '''returns a pipelined model'''
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
        return pipe

    def run(self):
        """set and train the pipeline"""

        # build pipeline
        pipeline = self.set_pipeline()

        # hold out
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, self.y, test_size=0.2)

        # train the pipeline
        pipeline.fit(self.X_train, self.y_train)

        return pipeline

    def evaluate(self):
        """evaluates the pipeline on df_test and return the RMSE"""
        pipeline = self.run()
        y_pred = pipeline.predict(self.X_val)
        rmse = compute_rmse(y_pred, self.y_val)
        return rmse

if __name__ == "__main__":
    # get data
    df = get_data()

    # clean data
    df = clean_data(df, test=False)

    # set X and y
    yy = df["fare_amount"]
    XX = df.drop("fare_amount", axis=1)

    # train
    trainer = Trainer(XX,yy)
    trainer.run()

    # evaluate
    rmse = trainer.evaluate()

    print('TODO')
