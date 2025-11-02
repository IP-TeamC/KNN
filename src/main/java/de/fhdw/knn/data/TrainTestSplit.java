package de.fhdw.knn.data;

public class TrainTestSplit {

    public DataSet train;
    public DataSet test;

    public TrainTestSplit(DataSet train, DataSet test) {
        this.train = train;
        this.test = test;
    }

}
