package de.fhdw.knn.util;

import lombok.SneakyThrows;

import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.BiConsumer;

public class FutureUtil {

    public static final ExecutorService EXECUTOR = Executors.newFixedThreadPool(12);

    public static void partitionedExecution(int n, int size, BiConsumer<Integer, Integer> f) {
        LinkedList<Future<?>> futures = new LinkedList<>();
        int[] partitions = FutureUtil.partition(n, size);
        for (int partition = 1; partition < partitions.length; partition++) {
            int first = partitions[partition - 1];
            int next = partitions[partition];
            Future<?> future = EXECUTOR.submit(() -> f.accept(first, next));
            futures.add(future);
        }
        FutureUtil.awaitFutures(futures);
    }

    public static int[] partition(int n, int size) {
        int partitionSize = size / n;
        int[] partitions = new int[n + 1];
        for (int i = 1; i < n; i++) {
            partitions[i] = partitions[i - 1] + partitionSize;
        }
        partitions[n] = size;
        return partitions;
    }

    @SneakyThrows
    public static void awaitFutures(List<Future<?>> futures) {
        for (Future<?> future : futures) {
            future.get();
        }
    }

}
