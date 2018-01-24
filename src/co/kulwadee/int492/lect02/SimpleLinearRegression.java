package co.kulwadee.int492.lect02;

public class SimpleLinearRegression {
    // dataset
    private double[] x; 
    private double[] y; 
    private int m;   // number of training examples
    private double maxX, minX;
    private double maxY, minY;

    // hyper-parameters
    private double learning_rate;
    private double tolerance;
    private int max_iteration;

    // model parameters
    private double theta0;
    private double theta1;

    public SimpleLinearRegression(double[] x, double[] y) {
        this.m = x.length;
        this.x = new double[m];
        this.y = new double[m];
        // keep min, max values for normalization
        minX = x[0]; maxX = x[0];
        minY = y[0]; maxY = y[0];
        for (int i = 1; i < m; i++) {
            if (minX > x[i]) minX = x[i];
            if (maxX < x[i]) maxX = x[i];

            if (minY > y[i]) minY = y[i];
            if (maxY < y[i]) maxY = y[i];
        }

        // copy normalized dataset for training
        for (int i = 0; i < m; i++) {
            this.x[i] = normalizeX(x[i]);
            this.y[i] = normalizeY(y[i]);
        }

        this.learning_rate = 0.01;
        this.tolerance = 1e-11;
        this.max_iteration = 10000;
        this.theta0 = 0.0;
        this.theta1 = 0.0; 
    }

    public double normalizeX(double x) {
        return (x-minX)/(maxX-minX);
    }

    public double normalizeY(double y) {
        return (y-minY)/(maxY-minY);
    }

    public double unnormalizeY(double normalizedY) {
        return (normalizedY * (maxY-minY)) + minY;
    }

    public double h_theta(double x) {
        return theta0 + theta1 * x;
    }

    public double deriveTheta1() {
        double sum = 0;
        int m = x.length;
        for (int i = 0; i < m; i++) {
            sum += (h_theta(x[i]) - y[i]) * x[i];
        }
        return (1.0/m)*sum;
    }

    public double deriveTheta0() {
        double sum = 0;
        int m = x.length;
        for (int i = 0; i < m; i++) {
            sum += h_theta(x[i]) - y[i];
        }
        return (1.0/m)*sum;
    }

    public double costFunction() {
        // squared error cost
        double sum = 0;
        int m = x.length;
        for (int i = 0; i < m; i++) {
            sum += Math.pow(h_theta(x[i]) - y[i], 2);
        }
        return sum/(2.0*m);
    }

    public void train() {
        int iters = 0;
        do {
            double dTheta1 = deriveTheta1();
            double dTheta0 = deriveTheta0();

            theta1 = theta1 - learning_rate * dTheta1;
            theta0 = theta0 - learning_rate * dTheta0;

            if (iters % 1000 == 0) {
                System.out.print(iters + ": ");
                printModel();
            }

            iters++;

        } while (iters < max_iteration && costFunction() > tolerance);
        System.out.println("training finished at iteration #: " + iters + ", cost: " + costFunction());
        System.out.println("Model: ");
        printModel();
    }

    public double predict(double size) {
        return unnormalizeY(h_theta(normalizeX(size)));
    }

    public void printModel() {
        System.out.println( String.format("price = %.4f + %.4f * x", theta0, theta1) );
    }

    public static void main(String[] args) {
        // normalized house dataset
        double[] x = {1076, 990, 1229, 731, 671, 1101, 909};
        double[] y = {398, 370, 425.9, 300, 312.1, 401, 383.4};
        /*
        double[] x = {0.73, 0.57, 1.00, 0.107, 0.01, 0.771, 0.426};
        double[] y = {0.810, 0.620, 0.98, 0.089, 0.0802, 0.815, 0.665};
        */

        // init. the model
        SimpleLinearRegression linreg = new SimpleLinearRegression(x, y);

        // train the model
        linreg.train();

        // predict price of a house
        double size = 750.0; 
        double price = linreg.predict(size);

        System.out.println(String.format("a house (%.1f m^2) is worth %.0f Baht", size, price));

    }
}
