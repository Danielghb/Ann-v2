// neural-net-tutorial.cpp
// David Miller, http://millermattson.com/dave
// See the associated video for instructions: http://vimeo.com/19569529


#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;

// Silly class to read training data from a text file -- Replace This.
// Replace class TrainingData with whatever you need to get input data into the
// program, e.g., connect to a database, or take a stream of data from stdin, or
// from a file specified by a command line argument, etc.

///*
class TrainingData
{
public:
    TrainingData(const string filename);
    bool isEof(void) { return m_trainingDataFile.eof(); }
    void getTopology(vector<unsigned> &topology);

    // Returns the number of input values read from the file:
    unsigned getNextInputs(vector<double> &inputVals);
    unsigned getTargetOutputs(vector<double> &targetOutputVals);

private:
    ifstream m_trainingDataFile;
};

void TrainingData::getTopology(vector<unsigned> &topology)
{
    string line;
    string label;

    getline(m_trainingDataFile, line);

    stringstream ss(line);

    ss >> label;

    if (this->isEof() || label.compare("topology:") != 0) {
            cout << "I'm broke!";
        abort();
    }

    while (!ss.eof()) {
        unsigned n;
        ss >> n;
        topology.push_back(n);
    }

    return;
}

TrainingData::TrainingData(const string filename)
{
    m_trainingDataFile.open(filename.c_str());
}

unsigned TrainingData::getNextInputs(vector<double> &inputVals)
{
    inputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss>> label;
    if (label.compare("in:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            inputVals.push_back(oneValue);
        }
    }

    return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputVals)
{
    targetOutputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss>> label;
    if (label.compare("out:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            targetOutputVals.push_back(oneValue);
        }
    }

    return targetOutputVals.size();
}
//*/

//http://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

/*
template <class T> const T& min (const T& a, const T& b) {
  return !(b<a)?a:b;     // or: return !comp(b,a)?a:b; for version (2)
}
*/

//RPROP
double DELTA_MIN = .000001;
double zeroTolerance = .0000000000000001;
double DEFAULT_MAX_STEP = 50;
double POSITIVE_ETA = 1.2;
double NEGATIVE_ETA = .5;

//https://github.com/encog/encog-c/blob/master/encog-core/rprop.c
/**
* Determine the sign of the value.
*
* @param value
* The value to check.
* @return -1 if less than zero, 1 if greater, or 0 if zero.
*/

int sign(double value) {
    if (fabs(value) < zeroTolerance) {
            return 0;
    } else if (value > 0) {
        return 1;
        } else {
            return -1;
            }
            }

struct Connection
{
    double weight;
    double deltaWeight;

    //RPROP
    double gradient = 0.0;
    double lastGradient = 0.0;
    double batchGradient = 0.0;
    int gradientCount = 0;
    double lastWeightChange = 0.0;
    double delta = 0.1;
    double lastDelta = 0.0;
};


class Neuron;

typedef vector<Neuron> Layer;

// ****************** class Neuron ******************
class Neuron
{
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    Neuron(unsigned numOutputs, unsigned myIndex, const vector<double> &weights);
    void setOutputVal(double val) { m_outputVal = val; }
    double getOutputVal(void) const { return m_outputVal; }
    void feedForward(const Layer &prevLayer);
    void calcOutputNodeDeltas(double targetVal);
    void calcHiddenNodeDeltas(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer, double lastError, double currentError);
    void updateInputWeights(Layer &prevLayer);

private:
    static double eta;   // [0.0..1.0] overall net training rate
    static double alpha; // [0.0..n] multiplier of last weight change (momentum)
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    static double randomWeight(void) { return rand() / double(RAND_MAX); }
    //double sumDOW(const Layer &nextLayer) const;
    double m_outputVal;
    vector<Connection> m_outputWeights;
    unsigned m_myIndex;
    double node_delta; //Jeff Heaton's Node Delta
};

double Neuron::eta = 0.15;    // overall net learning rate, [0.0..1.0]
double Neuron::alpha = 0.5;   // momentum, multiplier of last deltaWeight, [0.0..1.0]

//original
void Neuron::updateInputWeights(Layer &prevLayer)
{
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer

    //CURRENT NEURON
    cout << endl << "node delta: " << node_delta << endl << endl;

    for (unsigned n = 0; n < prevLayer.size(); ++n) {

        //PREVIOUS LAYER NEURON
        Neuron &neuron = prevLayer[n];

        //double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        //PREVIOUS LAYER NEURON OUTPUT * CURRENT NEURON's Node Delta
        double gradient = neuron.getOutputVal() * node_delta;

        //cout << "previous layer size: " << prevLayer.size() << endl;

        cout << neuron.getOutputVal() << endl << node_delta << endl;
        cout << "output value: " << neuron.getOutputVal() << endl;
        cout << "gradient: " <<  gradient << endl;

        //batch
        neuron.m_outputWeights[m_myIndex].batchGradient += gradient;
        neuron.m_outputWeights[m_myIndex].gradientCount ++;

        cout << "batch: " << neuron.m_outputWeights[m_myIndex].batchGradient << endl;

        cout << "count: " << neuron.m_outputWeights[m_myIndex].gradientCount << endl;

        //if count == 4, then it's  batch, and do RPROP
        if (neuron.m_outputWeights[m_myIndex].gradientCount == 4)
        {
            cout << "old batch: " << neuron.m_outputWeights[m_myIndex].lastGradient << endl;
            neuron.m_outputWeights[m_myIndex].gradient = neuron.m_outputWeights[m_myIndex].batchGradient;

            //reset
            neuron.m_outputWeights[m_myIndex].gradientCount = 0;
            neuron.m_outputWeights[m_myIndex].batchGradient = 0.0;

            int change = 0;

            change = sign (neuron.m_outputWeights[m_myIndex].gradient * neuron.m_outputWeights[m_myIndex].lastGradient);
            cout << "change: " << change << endl;

            double weightChange = 0.0;
            cout << "Old Delta: " << neuron.m_outputWeights[m_myIndex].delta << endl;

            if (change > 0)
            {
                neuron.m_outputWeights[m_myIndex].delta = min( neuron.m_outputWeights[m_myIndex].delta * POSITIVE_ETA , DEFAULT_MAX_STEP);
            }

            if (change < 0)
            {
                neuron.m_outputWeights[m_myIndex].delta = max( neuron.m_outputWeights[m_myIndex].delta * NEGATIVE_ETA , DELTA_MIN);
            }

            //section 2 (signer)
            cout << "Old Delta: " << neuron.m_outputWeights[m_myIndex].delta << endl;
            double newDeltaWeight = 0.0;


        //need this, otherwise it outputs -0 which has an effect when multiplied against neuron.m_outputtWeights[m_myIndex].delta

        /*
        if (neuron.m_outputWeights[m_myIndex].gradient == 0)
        {
            neuron.m_outputWeights[m_myIndex].gradient = 0;
        }
        */


            //int signer = 0.0;

            /*
            if (sign(neuron.m_outputWeights[m_myIndex].gradient) == 0)
            {
                //needed for -0 gradients
                if (neuron.m_outputWeights[m_myIndex].gradient == 0)
                {
                    signer = -1;
                }
                else
                {
                    signer = 1;
                }
            }
            else if (sign(neuron.m_outputWeights[m_myIndex].gradient) == 1)
            {
                signer = 1;
            }
            else if (sign(neuron.m_outputWeights[m_myIndex].gradient) == -1)
            {
                signer = -1;
            }
            */

            //section 3

            if (change == 0)
            {
                //cout << sign(neuron.m_outputWeights[m_myIndex].gradient) << "*" << neuron.m_outputWeights[m_myIndex].delta << endl;

                newDeltaWeight = sign(neuron.m_outputWeights[m_myIndex].gradient) * neuron.m_outputWeights[m_myIndex].delta;

            }
            else
                {
                    //cout << -sign(neuron.m_outputWeights[m_myIndex].gradient) << "*" << neuron.m_outputWeights[m_myIndex].delta << endl;
                    //newDeltaWeight = -sign(neuron.m_outputWeights[m_myIndex].gradient) * neuron.m_outputWeights[m_myIndex].delta;
                    newDeltaWeight = change * neuron.m_outputWeights[m_myIndex].delta;
                }




            //newDeltaWeight = -sign(gradient) * neuron.m_outputWeights[m_myIndex].delta;

            cout << "newDeltaWeight: " << newDeltaWeight << endl;
            cout << "Weight: " << neuron.m_outputWeights[m_myIndex].weight << endl;

            //section 4

            //neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
            neuron.m_outputWeights[m_myIndex].delta = newDeltaWeight;
            neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
            cout << "New Weight: " << neuron.m_outputWeights[m_myIndex].weight << endl;


            //section 5
            //update lastGradient (at end of processing of RPROP).
            neuron.m_outputWeights[m_myIndex].lastGradient = neuron.m_outputWeights[m_myIndex].gradient;

            }













/*
        double newDeltaWeight =
                // Individual input, magnified by the Node Delta and train rate:

                eta
                * gradient
                // Also add momentum = a fraction of the previous delta weight;
                + alpha
                * oldDeltaWeight;
*/
        system("pause");
    }
}

//includes true gradient
void Neuron::updateInputWeights(Layer &prevLayer, double lastError, double currentError)
{
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer

    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        Neuron &neuron = prevLayer[n];

        // multiply the current and previous gradient, and take the
        // sign. We want to see if the gradient has changed its sign.

        double weightChange = 0.0;
        int change = sign( neuron.m_outputWeights[m_myIndex].gradient * neuron.m_outputWeights[m_myIndex].lastGradient );

        neuron.m_outputWeights[m_myIndex].gradient =
        //double newDeltaWeight =
                // Individual input, magnified by the gradient and train rate:
                //eta
                //*
                neuron.getOutputVal()
                * node_delta;
                // Also add momentum = a fraction of the previous delta weight;
                //+ alpha
                //* oldDeltaWeight;

                cout << "gradient: " << neuron.m_outputWeights[m_myIndex].gradient << endl;

        // if the gradient has retained its sign, then we increase the
        // delta so that it will converge faster

        if (change > 0)
        {
            neuron.m_outputWeights[m_myIndex].delta = min( neuron.m_outputWeights[m_myIndex].delta * POSITIVE_ETA , DEFAULT_MAX_STEP);
            weightChange = -sign(neuron.m_outputWeights[m_myIndex].gradient) * neuron.m_outputWeights[m_myIndex].delta;
            neuron.m_outputWeights[m_myIndex].lastGradient = neuron.m_outputWeights[m_myIndex].gradient;
        }
        // if change<0, then the sign has changed, and the last
        // delta was too big

        else if (change < 0)
        {
            neuron.m_outputWeights[m_myIndex].delta = max( neuron.m_outputWeights[m_myIndex].delta * NEGATIVE_ETA , DELTA_MIN);

            if (currentError > lastError)
            {
                //weightChange = -neuron.m_outputWeights[m_myIndex].lastWeightChange;
            }
            neuron.m_outputWeights[m_myIndex].lastGradient = 0;
        }

        // if change==0 then there is no change to the delta
        else if (change == 0)
        {
            weightChange = -sign( neuron.m_outputWeights[m_myIndex].gradient ) * neuron.m_outputWeights[m_myIndex].delta;
            neuron.m_outputWeights[m_myIndex].lastGradient = neuron.m_outputWeights[m_myIndex].gradient;
        }

        neuron.m_outputWeights[m_myIndex].lastDelta = neuron.m_outputWeights[m_myIndex].delta;
        neuron.m_outputWeights[m_myIndex].lastWeightChange = weightChange;


        //weight is updated
        neuron.m_outputWeights[m_myIndex].weight += weightChange;
    }
}

/*
double Neuron::sumDOW(const Layer &nextLayer) const
{
    double sum = 0.0;

    // Sum our contributions of the errors at the nodes we feed.

    for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
        sum += m_outputWeights[n].weight * nextLayer[n].node_delta;
    }

    return sum;
}
*/

void Neuron::calcHiddenNodeDeltas(const Layer &nextLayer)
{
    double sum = 0.0;

    // Sum our contributions of the errors at the nodes we feed.
    // a single NEURON calculates the sum of the nodes it feeds
    // weighted error signal for all nodes
    // process weights going from current neuron to NEXT layer
    for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
        sum += m_outputWeights[n].weight * nextLayer[n].node_delta;
    }

    node_delta = sum * Neuron::transferFunctionDerivative(m_outputVal);

    //cout << "Hidden Node Delta: " << node_delta << endl;
}

void Neuron::calcOutputNodeDeltas(double targetVal)
{
    double delta = targetVal - m_outputVal;
    node_delta = delta * Neuron::transferFunctionDerivative(m_outputVal);
    //cout << "Output Node Delta: " << node_delta << endl;
}

double Neuron::transferFunction(double x)
{
    // tanh - output range [-1.0..1.0]

    //return tanh(x);
    return 1/(1+pow(exp(1),(-x)));
}

double Neuron::transferFunctionDerivative(double x)
{
    // tanh derivative
    //return (1.0 - x) * x;
    return (1.0 - x) * x;
}

void Neuron::feedForward(const Layer &prevLayer)
{
    double sum = 0.0;

    // Sum the previous layer's outputs (which are our inputs)
    // Include the bias node from the previous layer.

    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].getOutputVal() *
                prevLayer[n].m_outputWeights[m_myIndex].weight;
    }

    m_outputVal = Neuron::transferFunction(sum);
}

//default constructor
Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    for (unsigned c = 0; c < numOutputs; ++c) {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }

    m_myIndex = myIndex;
}

//default constructor that accepts weights
Neuron::Neuron(unsigned numOutputs, unsigned myIndex, const vector<double> &weights)
{
    for (unsigned c = 0; c < numOutputs; ++c) {
        m_outputWeights.push_back(Connection());
        //m_outputWeights.back().weight = randomWeight();
        m_outputWeights.back().weight = weights[c];
        cout << "inputted weight: " << weights[c] << endl;
    }

    m_myIndex = myIndex;
}

// ****************** class Net ******************
class Net
{
public:
    Net(const vector<unsigned> &topology);
    Net(const vector<unsigned> &topology, const vector<vector<double>> &weights);
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals) const;
    double getRecentAverageError(void) const { return m_recentAverageError; }
    double returnError();
    double returnLastError();

private:
    vector<Layer> m_layers; // m_layers[layerNum][neuronNum]
    double m_error;
    double m_last_error = 0.0;
    double m_recentAverageError;
    static double m_recentAverageSmoothingFactor;
};


double Net::m_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over


void Net::getResults(vector<double> &resultVals) const
{
    resultVals.clear();

    for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}

void Net::backProp(const vector<double> &targetVals)
{
    // Calculate overall net error (RMS of output neuron errors)

    Layer &outputLayer = m_layers.back();

    //backup last error
    m_last_error = m_recentAverageError;

    m_error = 0.0;

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
    m_error /= outputLayer.size() - 1; // get average error squared
    m_error = sqrt(m_error); // RMS

    // Implement a recent average measurement

    m_recentAverageError =
            (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
            / (m_recentAverageSmoothingFactor + 1.0);

    // Calculate output layer Node Deltas

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        outputLayer[n].calcOutputNodeDeltas(targetVals[n]);
    }

    // Calculate hidden layer Node Deltas
    // for each layer
    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        //for each neuron
        for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].calcHiddenNodeDeltas(nextLayer);
        }
    }

    // For all layers from outputs to first hidden layer,
    // update connection weights

    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];

        cout << endl << "layer: " << layerNum << endl;

        for (unsigned n = 0; n < layer.size() - 1; ++n) {
            layer[n].updateInputWeights(prevLayer);
            //layer[n].updateInputWeights(prevLayer, m_last_error, m_error);
        }
    }
}

double Net::returnError()
{
    return m_error;

}

double Net::returnLastError()
{
    return m_last_error;

}

void Net::feedForward(const vector<double> &inputVals)
{
    assert(inputVals.size() == m_layers[0].size() - 1);

    // Assign (latch) the input values into the input neurons
    for (unsigned i = 0; i < inputVals.size(); ++i) {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    // forward propagate
    for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
        Layer &prevLayer = m_layers[layerNum - 1];
        for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

//default constructor
Net::Net(const vector<unsigned> &topology)
{
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        m_layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

        // We have a new layer, now fill it with neurons, and
        // add a bias neuron in each layer.
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
            cout << "Made a Neuron!" << endl;
        }

        // Force the bias node's output to 1.0 (it was the last neuron pushed in this layer):
        m_layers.back().back().setOutputVal(1.0);
    }
}

//default constructor that accepts weights
Net::Net(const vector<unsigned> &topology, const vector<vector<double>> &weights)
{
    unsigned numLayers = topology.size();

    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        m_layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

        //cout << "layer: " << layerNum << endl;

        //# of layer Connections
        int layerConnections = topology[layerNum+1]*(topology[layerNum]-1);
        //cout << "layer connections: " << layerConnections << endl;

        //myWeights.push_back(weights[layerNum][]);

        // We have a new layer, now fill it with neurons, and
        // add a bias neuron in each layer.

        int connectionCounter = 0;

        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {

            vector <double> myWeights;

            for (int i = 0; i < numOutputs; i++)
            {
                myWeights.push_back(weights[layerNum][connectionCounter]);
                connectionCounter++;
            }

            m_layers.back().push_back(Neuron(numOutputs, neuronNum, myWeights));
            cout << "Made a Neuron!" << endl;
        }

        // Force the bias node's output to 1.0 (it was the last neuron pushed in this layer):
        m_layers.back().back().setOutputVal(1.0);
    }
}


void showVectorVals(string label, vector<double> &v)
{
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        cout << v[i] << " ";
    }

    cout << endl;
}


int main()
{

//section A

    // e.g., { 3, 2, 1 }
    vector<unsigned> topology;

    //nuerons by layer
    topology.push_back(2);
    topology.push_back(2);
    topology.push_back(1);

    //2d vector of weights
    vector<vector<double>> weights;

    for (int i = 1; i < topology.size(); i++)
    {
        weights.push_back(vector <double>());
    }

    //size of dimension
    cout << weights.size() << endl;

    //size of 1st dimension
    cout << weights[0].size() << endl;
    cout << weights[1].size() << endl;

    //1st layer [0,x]
    weights[0].push_back(-.06782947598673161);
    weights[0].push_back(.9487814395569221);
    weights[0].push_back(.22341077197888182);
    weights[0].push_back(.46158711646254);
    weights[0].push_back(-.4635107399577998);
    weights[0].push_back(.09750161997450091);


    //2nd layer [1,x]
    weights[1].push_back(-.22791948943117624);
    weights[1].push_back(.581714099641357);
    weights[1].push_back(.7792991203673414);

    cout << weights[0].size() << endl;
    cout << weights[1].size() << endl;


    Net myNet(topology, weights);

    TrainingData trainData("trainingData.txt");

    vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;



    //reads through data and implements training in online mode
    while (!trainData.isEof()) {
        ++trainingPass;
        cout << endl << "Pass " << trainingPass;

        // Get new input data and feed it forward:
        if (trainData.getNextInputs(inputVals) != topology[0]) {
            break;
        }
        showVectorVals(": Inputs:", inputVals);
        myNet.feedForward(inputVals);

        // Collect the net's actual output results:
        myNet.getResults(resultVals);
        showVectorVals("Outputs:", resultVals);

        // Train the net with what the outputs should have been:
        trainData.getTargetOutputs(targetVals);
        showVectorVals("Targets:", targetVals);
        assert(targetVals.size() == topology.back());

        myNet.backProp(targetVals);

        // Report how well the training is working, average over recent samples:
        cout << "Net recent average error: "
                << myNet.getRecentAverageError() << endl;
    }

    cout << endl << "Done" << endl;

/*
    vector <double> inputVals, resultVals, targetVals;

    inputVals.push_back(1);
    inputVals.push_back(0);

    targetVals.push_back(1);
    assert(targetVals.size() == topology.back());

    myNet.feedForward(inputVals);
    myNet.getResults(resultVals);

    cout << endl;
    showVectorVals("Outputs:", resultVals);

    //cout << "error: " << myNet.returnError() << endl;

    myNet.backProp(targetVals);

    // Report how well the training is working, average over recent samples:
        cout << endl << "Net recent average error: "
       /         << myNet.getRecentAverageError() << endl;
*/

//section b
/*
    TrainingData trainData("trainingData.txt");

    // e.g., { 3, 2, 1 }
    vector<unsigned> topology;
    trainData.getTopology(topology);

    Net myNet(topology);

    vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;


    while (!trainData.isEof()) {
        ++trainingPass;
        cout << endl << "Pass " << trainingPass;

        // Get new input data and feed it forward:
        if (trainData.getNextInputs(inputVals) != topology[0]) {
            break;
        }
        showVectorVals(": Inputs:", inputVals);
        myNet.feedForward(inputVals);

        // Collect the net's actual output results:
        myNet.getResults(resultVals);
        showVectorVals("Outputs:", resultVals);

        // Train the net what the outputs should have been:
        trainData.getTargetOutputs(targetVals);
        showVectorVals("Targets:", targetVals);
        assert(targetVals.size() == topology.back());

        myNet.backProp(targetVals);

        // Report how well the training is working, average over recent samples:
        cout << "Net recent average error: "
                << myNet.getRecentAverageError() << endl;
    }

    cout << endl << "Done" << endl;
*/
}

