package de.fhdw.knn.trainer.stop;

@FunctionalInterface
public interface StopFunction {

    StopFunction NEVER = loss -> false;

    boolean isFinished(double loss);

}
