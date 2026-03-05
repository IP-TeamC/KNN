package de.fhdw.knn.run;

import de.fhdw.knn.data.*;
import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.connection.WeightInitializer;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.trainer.Trainer;
import de.fhdw.knn.trainer.learningrate.ConstantLearningRate;
import de.fhdw.knn.trainer.loss.LossFunction;
import de.fhdw.knn.trainer.optimization.GradientDescent;
import de.fhdw.knn.trainer.optimization.OptimizationFunction;
import de.fhdw.knn.trainer.stop.EarlyStopping;
import de.fhdw.knn.trainer.stop.StopFunction;

import java.io.IOException;
import java.util.Arrays;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.IntStream;

class A22T {

    private static final String ALPHABET = "abcdefghijklmnopqrstuvwxyz ,:;-.";

    public static void main(String[] args) throws IOException {
        DataSet dataEncryption = CsvReader.readFile("data/a2_2_translation.csv", 1, line -> {
            String[] clearCipher = line.split("\",\"", 2);
            String clear = clearCipher[0].substring(1);
            String cipher = clearCipher[1].substring(0, clearCipher[1].length() - 1);
            return IntStream.range(0, clear.length()).mapToObj(i -> Map.entry(oneHot(clear.charAt(i)), oneHot(cipher.charAt(i))));
        });
        DataSet dataDecryption = new DataSet(dataEncryption.outputs, dataEncryption.inputs);

        Function<DataSet, Network> trainerFunction = A22T::trainNetwork;
        long trainStart = System.currentTimeMillis();
        final Network encryption = trainerFunction.apply(dataEncryption);
        final Network decryption = trainerFunction.apply(dataDecryption);
        long trainStop = System.currentTimeMillis();

        long trainTime = trainStop - trainStart;
        System.out.printf("Train Time: %d ms (%.2f s)%n", trainTime, trainTime / 1000.0);

        verify(encryption, decryption);
    }

    private static Network trainNetwork(final DataSet data) {
        DenseLayer[] denseLayers = DenseLayer.createLayers(ActivationFunction.SNAKE, ActivationFunction.SIGMOID, data.outputSize);
        Network network = new Network(42, WeightInitializer.GLOROT_UNIFORM, data.inputSize, denseLayers);

        LossFunction lossFunction = LossFunction.CROSS_ENTROPY_LOSS;
        StopFunction stopFunction = EarlyStopping.NEVER;
        OptimizationFunction optimizationFunction = new GradientDescent(lossFunction, new ConstantLearningRate(10));

        Trainer trainer = new Trainer(network, 1, false, 1, lossFunction, stopFunction, optimizationFunction);
        trainer.train(data);
        return network;
    }

    private static Network trainNetworkWithHidden(final DataSet data) {
        DenseLayer[] denseLayers = DenseLayer.createLayers(ActivationFunction.SNAKE, ActivationFunction.SIGMOID, 15, data.outputSize);
        Network network = new Network(42, WeightInitializer.GLOROT_UNIFORM, data.inputSize, denseLayers);

        LossFunction lossFunction = LossFunction.CROSS_ENTROPY_LOSS;
        StopFunction stopFunction = EarlyStopping.NEVER;
        OptimizationFunction optimizationFunction = new GradientDescent(lossFunction, new ConstantLearningRate(0.1));

        Trainer trainer = new Trainer(network, 10, true, 1, lossFunction, stopFunction, optimizationFunction);
        trainer.train(data);
        return network;
    }

    private static double[] oneHot(char c) {
        double[] oneHot = new double[ALPHABET.length()];
        oneHot[ALPHABET.indexOf(c)] = 1;
        return oneHot;
    }

    private static char oneHot(double[] oneHot) {
        int maxIndex = -1;
        double maxValue = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < oneHot.length; i++) {
            if (oneHot[i] > maxValue) {
                maxIndex = i;
                maxValue = oneHot[i];
            }
        }
        return ALPHABET.charAt(maxIndex);
    }

    private static String encrypt(final Network network, final String text) {
        double[][] encoded = text.chars().mapToObj(c -> oneHot((char) c)).toArray(double[][]::new);
        return Arrays.stream(network.predict(encoded))
                .map(A22T::oneHot)
                .reduce("", (prefix, c) -> prefix + c, (left, right) -> left + right);
    }

    private static void verify(final Network encryption, final Network decryption) {
        final String clear1 = "wenn wir ihm die vordersatze abgefragt haben und er sie zugegeben hat, mussen wir den schluss daraus nicht etwa auch noch fragen, sondern gradezu selbst ziehn: ja sogar wenn von den vordersatzen noch einer oder der andre fehlt, so nehmen wir ihn doch als gleichfalls eingeraumt an und ziehn den schluss. welches dann eine anwendung der fallacia non causae ut causae ist.";
        final String clear2 = "wenn man einen schluss machen will, so lasse man denselben nicht vorhersehn, sondern lasse sich unvermerkt die pramissen einzeln und zer- streut im gesprach zugeben, sonst wird der gegner allerhand schikanen versuchen; oder wenn zweifelhaft ist, dass der gegner sie zugebe, so stelle man die pramissen dieser pramissen auf; mache prosyllogismen ; lasse sich die pramissen mehrerer solcher prosyllogismen ohne ordnung durcheinander zugeben, also verdecke sein spiel, bis alles zugestanden ist, was man braucht. fuhre also die sache von weitem herbei. diese regeln gibt aristoteles. bedarf keines exempels.";
        final String clear3 = "die konsequenzmacherei. man erzwingt aus dem satze des gegners durch falsche folgerungen und verdrehung der begriffe satze, die nicht darin liegen und gar nicht die meinung des gegners sind, hingegen absurd oder gefahrlich sind: da es nun scheint, dass aus seinem satze solche satze, die entweder sich selbst oder anerkannten wahrheiten widersprechen, hervorgehn; so gilt dies fur eine indirekte widerlegung, apagoge: und ist wieder eine anwendung der fallacia non causae ut causae.";
        final String[] clear = new String[]{clear1, clear2, clear3};

        final long time1 = System.currentTimeMillis();
        final String encrypted1 = encrypt(encryption, clear1);
        final String encrypted2 = encrypt(encryption, clear2);
        final String encrypted3 = encrypt(encryption, clear3);
        final long time2 = System.currentTimeMillis();
        final String[] encrypted = new String[]{encrypted1, encrypted2, encrypted3};

        final String cipher1 = "zaii zoh ogc poa tuhpahsxfra xjkaqhxkf gxjai nip ah soa rnkakajai gxf, cnssai zoh pai sbgynss pxhxns iobgf afzx xnbg iubg qhxkai, suipahi khxparn sayjsf roagi: wx sukxh zaii tui pai tuhpahsxfrai iubg aoiah upah pah xipha qagyf, su iagcai zoh ogi pubg xys kyaobgqxyys aoikahxncf xi nip roagi pai sbgynss. zaybgas pxii aoia xizaipnik pah qxyyxbox iui bxnsxa nf bxnsxa osf.";
        final String cipher2 = "zaii cxi aoiai sbgynss cxbgai zoyy, su yxssa cxi paisayjai iobgf tuhgahsagi, suipahi yxssa sobg nitahcahlf poa ehxcossai aoirayi nip rah- sfhanf oc kasehxbg rnkajai, suisf zohp pah kakiah xyyahgxip sbgolxiai tahsnbgai; upah zaii rzaoqaygxqf osf, pxss pah kakiah soa rnkaja, su sfayya cxi poa ehxcossai poasah ehxcossai xnq; cxbga ehusdyyukoscai ; yxssa sobg poa ehxcossai caghahah suybgah ehusdyyukoscai ugia uhpinik pnhbgaoixipah rnkajai, xysu tahpabla saoi seoay, jos xyyas rnkasfxipai osf, zxs cxi jhxnbgf. qngha xysu poa sxbga tui zaofac gahjao. poasa hakayi kojf xhosfufayas. japxhq laoias avaceays.";
        final String cipher3 = "poa luisamnaircxbgahao. cxi ahrzoikf xns pac sxfra pas kakiahs pnhbg qxysbga quykahnikai nip tahphagnik pah jakhoqqa sxfra, poa iobgf pxhoi yoakai nip kxh iobgf poa caoinik pas kakiahs soip, goikakai xjsnhp upah kaqxghyobg soip: px as ini sbgaoif, pxss xns saoiac sxfra suybga sxfra, poa aifzapah sobg sayjsf upah xiahlxiifai zxghgaofai zopahsehabgai, gahtuhkagi; su koyf poas qnh aoia oipohalfa zopahyaknik, xexkuka: nip osf zoapah aoia xizaipnik pah qxyyxbox iui bxnsxa nf bxnsxa.";
        final String[] cipher = new String[]{cipher1, cipher2, cipher3};

        final long time3 = System.currentTimeMillis();
        final String decrypted1 = encrypt(decryption, cipher1);
        final String decrypted2 = encrypt(decryption, cipher2);
        final String decrypted3 = encrypt(decryption, cipher3);
        final long time4 = System.currentTimeMillis();
        final String[] decrypted = new String[]{decrypted1, decrypted2, decrypted3};

        System.out.println("\nVerification (Encryption):");
        boolean ok = compare(cipher, encrypted);
        System.out.println("\nVerification (Decryption):");
        ok = compare(clear, decrypted) && ok;
        System.out.println("\nOk: " + ok);
        System.out.println("Time: " + (time4 - time3 + time2 - time1) + " ms");
    }

    private static boolean compare(final String[] expected, final String[] predicted) {
        boolean ok = true;
        for (int i = 0; i < expected.length; i++) {
            System.out.println("\nExpected/Predicted:");
            System.out.println(expected[i]);
            System.out.println(predicted[i]);
            if (!expected[i].equals(predicted[i])) {
                System.out.println("Not equals: " + i);
                ok = false;
            } else {
                System.out.println("Equals: " + i);
            }
        }
        return ok;
    }

}
