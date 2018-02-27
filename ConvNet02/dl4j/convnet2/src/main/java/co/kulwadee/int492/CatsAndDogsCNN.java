package co.kulwadee.int492.convnet2;

import org.datavec.api.util.ClassPathResource;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.MultiImageTransform;
import org.datavec.image.transform.ResizeImageTransform;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.CropImageTransform;
import org.datavec.image.transform.ShowImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;

import org.deeplearning4j.util.ModelSerializer;

import org.apache.commons.io.FilenameUtils;
import java.io.File;
import java.util.Random;
import java.util.List;
import java.util.Arrays;
import java.util.ArrayList;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CatsAndDogsCNN {

    private static final Logger log = LoggerFactory.getLogger(CatsAndDogsCNN.class);
    private static int rngseed = 123;
    private static final Random randNumGen = new Random(rngseed);
    private static final int height = 150;
    private static final int width = 150;
    private static final int channels = 3;
    private static final int labelIndex = 1;
    private static final int batchSize = 16;
    private static final int iterations = 1;
    private static final int epochs = 80;

    public static void main(String[] args) throws Exception {

        log.info("Data load and pre-processing...");

        // Config data pre-processing
        // resize image to 150x150
        ImageTransform resizeTransform = new ResizeImageTransform(width, height);

        // flip and crop transformer for data augmentation
        List<ImageTransform> transforms = getTransformers();

        // setup data normalizer 
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

        // setup statistics GUI
        StatsStorage statsStorage = new InMemoryStatsStorage();
        UIServer uiServer = UIServer.getInstance();
        uiServer.attach(statsStorage);
                                                
        // Configure ImageRecordReader object
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);

        // laod training data
        FileSplit trainData = getTrainData();
        DataSetIterator trainDataIter = initTrainDataSetIterator(trainData, recordReader, resizeTransform, scaler);
        int numLabels = recordReader.numLabels();

        // Build Model
        log.info("Build convnet model...");
        MultiLayerNetwork convnet = buildConvNet(numLabels, statsStorage);

        log.info("Training on original dataset (resized to 150x150): num of Epoch = " + epochs);
        MultipleEpochsIterator trainIter = new MultipleEpochsIterator(epochs, trainDataIter);
        convnet.fit(trainIter);

        log.info("Training on augmented data", epochs);
        for (ImageTransform transform: transforms) {
            log.info("Training on transformation: " + transform.getClass().toString());

            trainDataIter = initTrainDataSetIterator(trainData, recordReader, transform, scaler);
            trainIter = new MultipleEpochsIterator(epochs, trainDataIter);

            convnet.fit(trainIter);

            FileSplit validationData = getValidationData();
            DataSetIterator validationDataIter = initTestDataSetIterator(validationData, recordReader, scaler);
            
            Evaluation eval = convnet.evaluate(validationDataIter);
            log.info(eval.stats(true));
        }

        log.info("Evaluate model...");
        FileSplit testData = getTestData();
        DataSetIterator testDataIter = initTestDataSetIterator(testData, recordReader, scaler);
        Evaluation eval = convnet.evaluate(testDataIter);
        log.info(eval.stats(true));

        log.info("Predict an example with the train model...");
        testDataIter.reset();
        DataSet testDataSet = testDataIter.next();
        List<String> allClassLabels = recordReader.getLabels();
        int labelIndex = testDataSet.getLabels().argMax(1).getInt(0);
        int[] predictedClasses = convnet.predict(testDataSet.getFeatures());
        String expectedResult = allClassLabels.get(labelIndex);
        String modelPrediction = allClassLabels.get(predictedClasses[0]);
        log.info("expected result is " + expectedResult + ", prediction is " +  modelPrediction);

        log.info("Save model...");
        String basePath = FilenameUtils.concat(System.getProperty("user.dir"), 
                                    "src/main/resources/cats_and_dogs_cnn.model1.bin");
        ModelSerializer.writeModel(convnet, basePath, true);

        log.info("Finished!..press Ctrl-C to end program.");
    }
    private static FileSplit getTrainData() throws java.io.FileNotFoundException {
        File parentDir = new ClassPathResource("data/cats_and_dogs_small/train").getFile();
        FileSplit trainData = new FileSplit(parentDir, BaseImageLoader.ALLOWED_FORMATS, randNumGen);
        return trainData;
    }
    private static FileSplit getValidationData() throws java.io.FileNotFoundException {
        File parentDir = new ClassPathResource("data/cats_and_dogs_small/validation").getFile();
        FileSplit testData = new FileSplit(parentDir, BaseImageLoader.ALLOWED_FORMATS, randNumGen);
        return testData;
    }
    private static FileSplit getTestData() throws java.io.FileNotFoundException {
        File parentDir = new ClassPathResource("data/cats_and_dogs_small/test").getFile();
        FileSplit testData = new FileSplit(parentDir, BaseImageLoader.ALLOWED_FORMATS, randNumGen);
        return testData;
    }
    private static List<ImageTransform> getTransformers() {
        // Data Augmentation - Flip, Crop
        MultiImageTransform flipTransform = new MultiImageTransform(
                                                    new FlipImageTransform(randNumGen),
                                                    new ResizeImageTransform(width, height));
        MultiImageTransform cropTransform = new MultiImageTransform(
                                                    new CropImageTransform(30),
                                                    new ResizeImageTransform(width, height));
        List<ImageTransform> transforms = Arrays.asList(new ImageTransform[]{flipTransform, cropTransform});

        return transforms;
    }
    private static DataSetIterator initTrainDataSetIterator(FileSplit dataset,
                                    ImageRecordReader recordReader, ImageTransform transformer, 
                                    DataNormalization scaler) throws java.io.IOException {
        recordReader.initialize(dataset, transformer);
        int numLabels = recordReader.numLabels();
        DataSetIterator dataIter = new RecordReaderDataSetIterator(
                                        recordReader, batchSize, labelIndex, numLabels);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        return dataIter;
    }
    private static DataSetIterator initTestDataSetIterator(FileSplit dataset, 
                                    ImageRecordReader recordReader, DataNormalization scaler) 
                                    throws java.io.IOException {
        recordReader.initialize(dataset);
        int numLabels = recordReader.numLabels();
        DataSetIterator dataIter = new RecordReaderDataSetIterator(
                                        recordReader, batchSize, labelIndex, numLabels);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        return dataIter;
    }
    private static MultiLayerNetwork buildConvNet(int numLabels, StatsStorage statsStorage) {
        //log.info("Configure CONVNET..");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(rngseed)
            .iterations(iterations)
            .learningRate(.01)
            .weightInit(WeightInit.XAVIER)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Updater.RMSPROP)
            .list()
            .layer(0, new ConvolutionLayer.Builder(3, 3)
                    .nIn(channels)
                    .stride(1, 1)
                    .nOut(32)
                    .activation(Activation.RELU)
                    .build())
            .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build())
            .layer(2, new ConvolutionLayer.Builder(3, 3)
                    .stride(1, 1)
                    .nOut(64)
                    .activation(Activation.RELU)
                    .build())
            .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build())
            .layer(4, new ConvolutionLayer.Builder(3, 3)
                    .stride(1, 1)
                    .nOut(128)
                    .activation(Activation.RELU)
                    .build())
            .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build())
            .layer(6, new ConvolutionLayer.Builder(3, 3)
                    .stride(1, 1)
                    .nOut(128)
                    .activation(Activation.RELU)
                    .build())
            .layer(7, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build())
            .layer(8, new DropoutLayer.Builder(0.3)
                    .build())
            .layer(9, new DenseLayer.Builder()
                    .activation(Activation.RELU)
                    .nOut(512)
                    .build())
            .layer(10, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nOut(numLabels)
                    .activation(Activation.SOFTMAX)
                    .build())
            .setInputType(InputType.convolutionalFlat(height, width, channels))
            .backprop(true).pretrain(false).build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        System.out.println(net.summary());

        net.setListeners((IterationListener)new StatsListener(statsStorage), 
                            new ScoreIterationListener(iterations));

        return net;
    }
}
