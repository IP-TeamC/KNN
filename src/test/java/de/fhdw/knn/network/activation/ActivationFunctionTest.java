package de.fhdw.knn.network.activation;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class ActivationFunctionTest {

    @Test
    public void testLinear() {
        ActivationFunction linear = ActivationFunction.LINEAR;
        assertEquals(5, linear.calc(5));
        assertEquals(10, linear.calc(10));
        assertEquals(-3.5, linear.calc(-3.5));
        assertEquals(0, linear.calc(0));

        assertEquals(1, linear.derived(5, 5));
        assertEquals(1, linear.derived(10, 10));
        assertEquals(1, linear.derived(-3.5, -3.5));
        assertEquals(1, linear.derived(0, 0));
    }

    @Test
    public void testReLU() {
        ActivationFunction relu = ActivationFunction.RELU;
        assertEquals(5, relu.calc(5));
        assertEquals(10, relu.calc(10));
        assertEquals(0, relu.calc(-3.5));
        assertEquals(0, relu.calc(0));

        assertEquals(1, relu.derived(5, 5));
        assertEquals(1, relu.derived(10, 10));
        assertEquals(0, relu.derived(-3.5, -3.5));
        assertEquals(0, relu.derived(0, 0));
    }

    @Test
    public void testSigmoid() {
        ActivationFunction sigmoid = ActivationFunction.SIGMOID;
        assertEquals(0.993, sigmoid.calc(5), 1e-3);
        assertEquals(0.622, sigmoid.calc(0.5), 1e-3);
        assertEquals(0.524, sigmoid.calc(0.1), 1e-3);
        assertEquals(0.029, sigmoid.calc(-3.5), 1e-3);
        assertEquals(0.413, sigmoid.calc(-0.35), 1e-3);
        assertEquals(0.5, sigmoid.calc(0), 1e-3);

        assertEquals(0.007, sigmoid.derived(5, 0.993), 1e-3);
        assertEquals(0.235, sigmoid.derived(0.5, 0.622), 1e-3);
        assertEquals(0.249, sigmoid.derived(0.1, 0.524), 1e-3);
        assertEquals(0.028, sigmoid.derived(-3.5, 0.029), 1e-3);
        assertEquals(0.242, sigmoid.derived(-0.35, 0.413), 1e-3);
        assertEquals(0.25, sigmoid.derived(0, 0.5), 1e-3);
    }

    @Test
    public void testSinus() {
        ActivationFunction sin = ActivationFunction.SIN;
        assertEquals(-0.959, sin.calc(5), 1e-3);
        assertEquals(-0.544, sin.calc(10), 1e-3);
        assertEquals(0.351, sin.calc(-3.5), 1e-3);
        assertEquals(0, sin.calc(0), 1e-3);

        assertEquals(0.284, sin.derived(5, -0.959), 1e-3);
        assertEquals(-0.839, sin.derived(10, -0.544), 1e-3);
        assertEquals(-0.936, sin.derived(-3.5, 0.351), 1e-3);
        assertEquals(1, sin.derived(0, 0), 1e-3);
    }

    @Test
    public void testSnake() {
        ActivationFunction snake = ActivationFunction.SNAKE;
        assertEquals(5.920, snake.calc(5), 1e-3);
        assertEquals(10.296, snake.calc(10), 1e-3);
        assertEquals(-3.377, snake.calc(-3.5), 1e-3);
        assertEquals(0, snake.calc(0), 1e-3);

        assertEquals(0.456, snake.derived(5, 5.920), 1e-3);
        assertEquals(1.913, snake.derived(10, 10.296), 1e-3);
        assertEquals(0.343, snake.derived(-3.5, -3.377), 1e-3);
        assertEquals(1, snake.derived(0, 0), 1e-3);
    }

    @Test
    public void testSoftplus() {
        ActivationFunction softplus = ActivationFunction.SOFTPLUS;
        assertEquals(5.007, softplus.calc(5), 1e-3);
        assertEquals(10, softplus.calc(10), 1e-3);
        assertEquals(0.030, softplus.calc(-3.5), 1e-3);
        assertEquals(0.693, softplus.calc(0), 1e-3);

        assertEquals(0.993, softplus.derived(5, 5.007), 1e-3);
        assertEquals(0.999, softplus.derived(10, 10), 1e-3);
        assertEquals(0.029, softplus.derived(-3.5, 0.030), 1e-3);
        assertEquals(0.5, softplus.derived(0, 0), 1e-3);
    }

    @Test
    public void testSwish() {
        ActivationFunction softplus = ActivationFunction.SWISH;
        assertEquals(4.967, softplus.calc(5), 1e-3);
        assertEquals(10, softplus.calc(10), 1e-3);
        assertEquals(-0.103, softplus.calc(-3.5), 1e-3);
        assertEquals(0, softplus.calc(0), 1e-3);

        assertEquals(1.026, softplus.derived(5, 4.967), 1e-3);
        assertEquals(1.000, softplus.derived(10, 10), 1e-3);
        assertEquals(-0.070, softplus.derived(-3.5, -0.103), 1e-3);
        assertEquals(0.5, softplus.derived(0, 0), 1e-3);
    }

    @Test
    public void testTanh() {
        ActivationFunction tanh = ActivationFunction.TANH;
        assertEquals(1, tanh.calc(5), 1e-3);
        assertEquals(0.462, tanh.calc(0.5), 1e-3);
        assertEquals(1, tanh.calc(10), 1e-3);
        assertEquals(0.100, tanh.calc(0.1), 1e-3);
        assertEquals(-0.998, tanh.calc(-3.5), 1e-3);
        assertEquals(-0.336, tanh.calc(-0.35), 1e-3);
        assertEquals(0, tanh.calc(0), 1e-3);

        assertEquals(0, tanh.derived(5, 1), 1e-3);
        assertEquals(0.786, tanh.derived(0.5, 0.462), 1e-3);
        assertEquals(0, tanh.derived(10, 1), 1e-3);
        assertEquals(0.990, tanh.derived(10, 0.1), 1e-3);
        assertEquals(0.004, tanh.derived(-3.5, -0.998), 1e-3);
        assertEquals(0.887, tanh.derived(-0.35, -0.336), 1e-3);
        assertEquals(1, tanh.derived(0, 0), 1e-3);
    }

}
