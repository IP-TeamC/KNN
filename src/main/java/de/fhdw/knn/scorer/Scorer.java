package de.fhdw.knn.scorer;

import de.fhdw.knn.data.DataSet;

public interface Scorer {

    Score score(DataSet data);

}
