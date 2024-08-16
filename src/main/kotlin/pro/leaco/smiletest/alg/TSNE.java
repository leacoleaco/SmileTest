package pro.leaco.smiletest.alg;

import me.tongfei.progressbar.ProgressBar;
import me.tongfei.progressbar.ProgressBarBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import smile.math.MathEx;
import smile.stat.distribution.GaussianDistribution;

import java.io.Serializable;
import java.util.Arrays;
import java.util.stream.IntStream;

public class TSNE implements Serializable {
    private static final long serialVersionUID = 2L;
    private static final Logger logger = LoggerFactory.getLogger(TSNE.class);
    public final double[][] coordinates;
    private final double eta;
    private int totalIter;
    private double momentum;
    private final double finalMomentum;
    private final int momentumSwitchIter;
    private final double minGain;
    private final double[][] gains;
    private final double[][] P;
    private final double[][] Q;
    private double Qsum;
    private double cost;

    public TSNE(double[][] X, int d) {
        this(X, d, 20.0, 200.0, 1000);
    }

    public TSNE(double[][] X, int d, double perplexity, double eta, int iterations) {
        this.totalIter = 0;
        this.momentum = 0.5;
        this.finalMomentum = 0.8;
        this.momentumSwitchIter = 250;
        this.minGain = 0.01;
        this.eta = eta;
        int n = X.length;
        double[][] D;
        if (X.length == X[0].length) {
            D = X;
        } else {
            D = new double[n][n];
            MathEx.pdist(X, D, MathEx::squaredDistance);
        }

        this.coordinates = new double[n][d];
        double[][] Y = this.coordinates;
        this.gains = new double[n][d];
        GaussianDistribution gaussian = new GaussianDistribution(0.0, 1.0E-4);


        ProgressBar bar = new ProgressBarBuilder()
                .setTaskName("Initializing")
                .setInitialMax(n)
                .build();
        for (int i = 0; i < n; ++i) {
            bar.step();
            Arrays.fill(this.gains[i], 1.0);
            double[] Yi = Y[i];

            for (i = 0; i < d; ++i) {
                Yi[i] = gaussian.rand();
            }
        }
        bar.close();

        this.P = this.expd(D, perplexity, 0.001);
        this.Q = new double[n][n];
        double Psum = (double) (2 * n);

        for (int i = 0; i < n; ++i) {
            double[] Pi = this.P[i];

            for (int j = 0; j < i; ++j) {
                double p = 12.0 * (Pi[j] + this.P[j][i]) / Psum;
                if (Double.isNaN(p) || p < 1.0E-16) {
                    p = 1.0E-16;
                }

                Pi[j] = p;
                this.P[j][i] = p;
            }
        }

        this.update(iterations);
    }

    public double cost() {
        return this.cost;
    }

    public void update(int iterations) {
        double[][] Y = this.coordinates;
        int n = Y.length;
        int d = Y[0].length;
        double[][] dY = new double[n][d];
        double[][] dC = new double[n][d];


        ProgressBar bar = new ProgressBarBuilder()
                .setInitialMax(iterations)
                .build();

        for (int iter = 1; iter <= iterations; ++this.totalIter) {
            bar.step();
            this.Qsum = this.computeQ(Y, this.Q);
            IntStream.range(0, n).parallel().forEach((ix) -> {
                this.sne(ix, dY[ix], dC[ix]);
            });
            IntStream.range(0, n).parallel().forEach((ix) -> {
                double[] Yi = Y[ix];
                double[] dYi = dY[ix];
                double[] dCi = dC[ix];
                double[] g = this.gains[ix];

                for (int k = 0; k < d; ++k) {
                    dYi[k] = this.momentum * dYi[k] - this.eta * g[k] * dCi[k];
                    Yi[k] += dYi[k];
                }

            });
            if (this.totalIter == 250) {
                this.momentum = 0.8;

                for (int i = 0; i < n; ++i) {
                    double[] Pi = this.P[i];

                    for (int j = 0; j < n; ++j) {
                        Pi[j] /= 12.0;
                    }
                }
            }

            if (iter % 100 == 0) {
                this.cost = this.computeCost(this.P, this.Q);
                logger.info("Error after {} iterations: {}", iter, this.cost);
            }

            ++iter;
        }

        bar.close();

        double[] colMeans = MathEx.colMeans(Y);
        IntStream.range(0, n).parallel().forEach((ix) -> {
            double[] Yi = Y[ix];

            for (int j = 0; j < d; ++j) {
                Yi[j] -= colMeans[j];
            }

        });
        if (iterations % 100 != 0) {
            this.cost = this.computeCost(this.P, this.Q);
            logger.info("Error after {} iterations: {}", iterations, this.cost);
        }

    }

    private void sne(int i, double[] dY, double[] dC) {
        double[][] Y = this.coordinates;
        int n = Y.length;
        int d = Y[0].length;
        double[] Yi = Y[i];
        double[] Pi = this.P[i];
        double[] Qi = this.Q[i];
        double[] g = this.gains[i];
        Arrays.fill(dC, 0.0);

        int k;
        for (k = 0; k < n; ++k) {
            if (i != k) {
                double[] Yj = Y[k];
                double q = Qi[k];
                double z = (Pi[k] - q / this.Qsum) * q;

                for (int j = 0; j < d; ++j) {
                    dC[j] += 4.0 * (Yi[j] - Yj[j]) * z;
                }
            }
        }

        for (k = 0; k < d; ++k) {
            g[k] = Math.signum(dC[k]) != Math.signum(dY[k]) ? g[k] + 0.2 : g[k] * 0.8;
            if (g[k] < 0.01) {
                g[k] = 0.01;
            }
        }

    }

    private double[][] expd(double[][] D, double perplexity, double tol) {
        int n = D.length;
        double[][] P = new double[n][n];
        double[] DiSum = MathEx.rowSums(D);
        IntStream.range(0, n).parallel().forEach((i) -> {
            double logU = MathEx.log2(perplexity);
            double[] Pi = P[i];
            double[] Di = D[i];
            double beta = Math.sqrt((double) (n - 1) / DiSum[i]);
            double betamin = 0.0;
            double betamax = Double.POSITIVE_INFINITY;
            logger.debug("initial beta[{}] = {}", i, beta);
            double Hdiff = Double.MAX_VALUE;

            for (int iter = 0; Math.abs(Hdiff) > tol && iter < 50; ++iter) {
                double Pisum = 0.0;
                double H = 0.0;

                int j;
                for (j = 0; j < n; ++j) {
                    double d = beta * Di[j];
                    double p = Math.exp(-d);
                    Pi[j] = p;
                    Pisum += p;
                    H += p * d;
                }

                Pi[i] = 0.0;
                --Pisum;
                H = MathEx.log2(Pisum) + H / Pisum;
                Hdiff = H - logU;
                if (Math.abs(Hdiff) > tol) {
                    if (Hdiff > 0.0) {
                        betamin = beta;
                        if (Double.isInfinite(betamax)) {
                            beta *= 2.0;
                        } else {
                            beta = (beta + betamax) / 2.0;
                        }
                    } else {
                        betamax = beta;
                        beta = (beta + betamin) / 2.0;
                    }
                } else {
                    for (j = 0; j < n; ++j) {
                        Pi[j] /= Pisum;
                    }
                }

                logger.debug("Hdiff = {}, beta[{}] = {}, H = {}, logU = {}", new Object[]{Hdiff, i, beta, H, logU});
            }

        });
        return P;
    }

    private double computeQ(double[][] Y, double[][] Q) {
        int n = Y.length;
        double[] rowSum = IntStream.range(0, n).parallel().mapToDouble((i) -> {
            double[] Yi = Y[i];
            double[] Qi = Q[i];
            double sum = 0.0;

            for (int j = 0; j < n; ++j) {
                double q = 1.0 / (1.0 + MathEx.squaredDistance(Yi, Y[j]));
                Qi[j] = q;
                sum += q;
            }

            return sum;
        }).toArray();
        return MathEx.sum(rowSum);
    }

    private double computeCost(double[][] P, double[][] Q) {
        return 2.0 * IntStream.range(0, Q.length).parallel().mapToDouble((i) -> {
            double[] Pi = P[i];
            double[] Qi = Q[i];
            double C = 0.0;

            for (int j = 0; j < i; ++j) {
                double p = Pi[j];
                double q = Qi[j] / this.Qsum;
                if (Double.isNaN(q) || q < 1.0E-16) {
                    q = 1.0E-16;
                }

                C += p * MathEx.log2(p / q);
            }

            return C;
        }).sum();
    }
}
