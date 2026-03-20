package de.fhdw.knn.config.scorer;

import de.fhdw.knn.scorer.ClassificationScorer;
import de.fhdw.knn.scorer.Scorer;
import de.fhdw.knn.trainer.loss.LossFunction;
import de.fhdw.knn.util.TestUtil;
import org.junit.jupiter.api.Test;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

class ConfScorerTest {

    @Test
    void testConfScorerCreateClassificationScorer() {
        ConfScorer conf = new ConfScorer();
        conf.type = "ClassificationScorer";

        Optional<Scorer> scorer = conf.create(TestUtil.simpleDummyNetwork());

        assertTrue(scorer.isPresent());
        assertInstanceOf(ClassificationScorer.class, scorer.get());
    }

    @Test
    void testConfScorerCreateUnknownTypeThrows() {
        ConfScorer conf = new ConfScorer();
        conf.type = "UnknownScorer";

        assertThrows(IllegalArgumentException.class, () -> conf.create(TestUtil.simpleDummyNetwork()));
    }


    @Test
    void testConfScorerCreateTypeNull() {
        ConfScorer conf = new ConfScorer();
        conf.type = null;

        Optional<Scorer> scorer = conf.create(TestUtil.simpleDummyNetwork());
        assertTrue(scorer.isEmpty());
    }

    @Test
    void testConfScorerGetLossFunction() {
        ConfScorer conf = new ConfScorer();
        conf.lossFunction = "MEAN_SQUARED_ERROR";

        Optional<LossFunction> lossFunction = conf.getLossFunction();
        assertTrue(lossFunction.isPresent());
    }

    @Test
    void testConfScorerGetLossFunctionNull() {
        ConfScorer conf = new ConfScorer();
        conf.lossFunction = null;

        Optional<LossFunction> lossFunction = conf.getLossFunction();
        assertTrue(lossFunction.isEmpty());
    }
}