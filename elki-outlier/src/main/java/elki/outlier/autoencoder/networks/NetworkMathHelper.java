package elki.outlier.autoencoder.networks;

import elki.math.linearalgebra.VMath;

import static elki.math.linearalgebra.VMath.*;

public class NetworkMathHelper {
    public static double[] ReLuGradient(double[] inputValues, double[] upstreamGradient){
        double[] localGradient = new double[inputValues.length];
        for(int i = 0; i < localGradient.length; i++){
            if(inputValues[i] > 0){
                localGradient[i] = upstreamGradient[i];
            }
        }
        return localGradient;
    }

    public static double[][] RandLayerWeightGradient(double[] inputValues, double[] upstreamGradient, boolean[][] activeConnections){
        double[][] weightGradient = VMath.timesTranspose(upstreamGradient, inputValues);
        for(int i = 0; i < activeConnections.length; i++){
            for(int j = 0; j < activeConnections[0].length; j++){
                if(!activeConnections[i][j]){
                    weightGradient[j][i] = 0;
                }
            }
        }
        return weightGradient;
    }

    public static double[] ReLu(double[] input){
        double[] output = new double[input.length];
        for(int i = 0; i < input.length; i++){
            output[i] = Math.max(0, input[i]);
        }
        return output;
    }

    /*
    Calculates sigmoid activation function for a vector inplace.
     */
    public static double[] sigmoid(double[] input){
        for (int i = 0; i < input.length; i++){
            input[i] = sigmoid(input[i]);
        }
        return input;
    }

    public static double sigmoid(double a){
        return 1.0/ (1.0 + Math.exp(a));
    }

    public static double[] sigmoidGradient(final double[] sigmoidOutput){
        final double[] grad = new double[sigmoidOutput.length];
        for (int i = 0; i < sigmoidOutput.length; i++){
            grad[i] = sigmoidOutput[i] * (1 - sigmoidOutput[i]);
        }
        return grad;
    }

    public static double[] root(final double[] v1, final double eps){
        final double[] v = new double[v1.length];
        for (int i = 0; i < v1.length; i++){
            v[i] = Math.sqrt(v1[i] + eps);
        }
        return v;
    }

    public static double[] divideEquals(final double s1, final double[] v1){
        for(int i = 0; i < v1.length; i++){
            v1[i] = s1 / v1[i];
        }
        return v1;
    }

    public static double[] hadamardSquare(final double[] v1){
        final double[] v = new double[v1.length];
        for(int i = 0;i < v1.length; i++){
            v[i] = v1[i] * v1[i];
        }
        return v;
    }

    /*
    Hadamard product of a transposed Matrix with itself: m = m1^T * m1^T
     */

    public static double[][] transposeHadamardSquare(final double[][] m1){
        final int rowdim = m1.length, coldim = getColumnDimensionality(m1);
        final double[][] m = new double[coldim][rowdim];
        for(int i = 0; i < rowdim; i++){
            for (int j = 0; j < coldim; j++){
                m[j][i] = m1[i][j] * m1[i][j];
            }
        }
        return m;
    }
}
