package elki.outlier.autoencoder.networks;

import elki.math.linearalgebra.VMath;

import static elki.math.linearalgebra.VMath.*;

public class NetworkMathHelper {

    /**Calculates the downstream gradient of a ReLu function given the upstream gradient and the input values of ReLu.
     * <code>dL/dx = dL/dy * dy/dx</code>
     *
     * @param inputValues Input vector for ReLu
     * @param upstreamGradient Upstream gradient
     * @return downstream gradient
     */
    public static double[] ReLuGradient(double[] inputValues, double[] upstreamGradient){
        double[] localGradient = new double[inputValues.length];
        for(int i = 0; i < localGradient.length; i++){
            if(inputValues[i] > 0){
                localGradient[i] = upstreamGradient[i];
            }
        }
        return localGradient;
    }

    /**
     * Calcualtes the downstream gradient of the weight <code> W </code> of an (almost) fully connected layer <code>y = W x</code>
     * @param inputValues Input values <code> x </code>
     * @param upstreamGradient upstream Gradient <code>dy</code>
     * @param activeConnections Bitmap which connections of the network are active
     * @return <code>dW = dy^T x</code>, accounted for inactive connections
     */
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

    /**
     * Calculates the vector valued ReLu function <code>ReLu(a) = max(0, a)</code>
     */
    public static double[] ReLu(double[] input){
        double[] output = new double[input.length];
        for(int i = 0; i < input.length; i++){
            output[i] = Math.max(0, input[i]);
        }
        return output;
    }

    /**
    Calculates sigmoid activation function <code>s(a) = 1/(1 + exp(a))</code> for a vector <bold>inplace</bold>.
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

    /**
     * Calculates the vector valued gradient of the sigmoid function <code>s(a) = 1 / (1 + exp (a))</code> given
     * <code>s(a)</code> by <code>s(a) * (1 - s(a))</code>
     * @param sigmoidOutput The output of the sigmoid function
     * @return The gradient of the sigmoid function.
     */
    public static double[] sigmoidGradient(final double[] sigmoidOutput){
        final double[] grad = new double[sigmoidOutput.length];
        for (int i = 0; i < sigmoidOutput.length; i++){
            grad[i] = sigmoidOutput[i] * (1 - sigmoidOutput[i]);
        }
        return grad;
    }

    /**
     * Calcualtes the element-wise root of a vector <code>v1</code>. For stability, a <code>eps</code> value is added to each element.
     * @param v1 Value vector
     * @param eps Eps value
     * @return <code>root(v1 + eps)</code>
     */
    public static double[] root(final double[] v1, final double eps){
        final double[] v = new double[v1.length];
        for (int i = 0; i < v1.length; i++){
            v[i] = Math.sqrt(v1[i] + eps);
        }
        return v;
    }

    /**
     * Hadamard product of a vector with itself: v = v1 * v1
     * @param v1 Value Vector
     * @return v
     */
    public static double[] hadamardSquare(final double[] v1){
        final double[] v = new double[v1.length];
        for(int i = 0;i < v1.length; i++){
            v[i] = v1[i] * v1[i];
        }
        return v;
    }
    /**
     * Hadamard product of a matrix with itself: m = m1 * m1
     * @param m1 Value Matrix
     */
    public static double[][] hadamardSquare(final double[][] m1){
        final int rowdim = m1.length, coldim = getColumnDimensionality(m1);
        final double[][] m = new double[rowdim][coldim];
        for(int i = 0; i < rowdim; i++){
            for (int j = 0; j < coldim; j++){
                m[i][j] = m1[i][j] * m1[i][j];
            }
        }
        return m;
    }


    /**
    Hadamard product of a transposed Matrix with itself: m = m1^T * m1^T
     * @param m1 Value Matrix
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

    /**
     * Clips a matrix with bounds inplace, for all values x : lower leq x leq upper
     * @param m1 Value matrix
     * @param upper upper bound
     * @param lower lower bound
     * @return
     */
    public static double[][] clip(final double[][] m1, double upper, double lower){
        final int rowdim = m1.length, coldim = getColumnDimensionality(m1);
        for (int i = 0; i < rowdim; i++){
            for (int j = 0; j < coldim; j++){
                m1[i][j] = Math.min(m1[i][j], upper);
                m1[i][j] = Math.max(m1[i][j], lower);
            }
        }
        return m1;
    }
}
