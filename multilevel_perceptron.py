import math

ETA = 0.1
EPOCH = 1000

def weighted_sum(X,W):
    return sum([x*w for x,w in zip(X,W)])

def signoid(x):
    return (1/(1+math.exp(-1*x)))

def layar_output(inputs,weights):
    return weighted_sum(inputs,weights)

def error_measure(outputs,targets):
    return sum(((t-o)**2 for t,o in zip(targets,outputs))) / (len(targets)*2)

def delta_output_layer(outputs,targets):
    return [o*(1-o)*(t-o) for t,o in zip(targets,outputs)]

def delta_hidden_layer(outputs,downstream_weights, downstream_deltas):
    return [[o*(1-o)*downstream_weights[j]*downstream_deltas[i] for o,j in zip(outputs[i],range(3))] for i in range(4)]

def update_weights(weights,deltas, inputs):
    return [weights[j]+(ETA*deltas*x) for x,j in zip(inputs,range(3))]


# main code starts here
inputs = [[0, 0, 1],
          [0, 1, 1],
          [1, 0, 1],
          [1, 1, 1]]

targets = [0, 1, 1, 0]

number_of_inputs=int(len(inputs))

weight_layer_1 = [[0.15, 0.20, 0.35],
                 [0.25, 0.30, 0.35]]

weight_layer_2 = [0.35, 0.40, 0.45]

iteration = 0
y=[]
y_list=[]
delta_hidden_list=[[]]
delta_output_list=[]
error=0
hidden_layer_output=[]
outputs=[]


while iteration<=EPOCH:
    iteration+=1
    for i in range(number_of_inputs):
        y.append(signoid(layar_output(inputs[i],weight_layer_1[0])))                   #Calculate hidden  1st neuron output
        y.append(signoid(layar_output(inputs[i],weight_layer_1[1])))                   #Calculate hidden  2nd neuron output
        y.append(1)                                                                    #3rd neuron output

        y_list.append(y.copy())                                                        #Store hidden neurons outputs

        outputs.append(signoid(layar_output(y,weight_layer_2)))
        y.clear()


    error=error_measure(outputs,targets)                                               #Error Calculate
    delta_output_list=delta_output_layer(outputs,targets)                              #Calculate Output layer delta value
    delta_hidden_list=delta_hidden_layer(y_list,weight_layer_2,delta_output_list)      #Calculate Hidden layer delta

    # Weight update for weight layer 2
    for i in range(4):
        weight_layer_2=update_weights(weight_layer_2,delta_output_list[i],y_list[i])

    # Wright update for weight layer 1
    for i in range(4):
         for j in range(2):
            weight_layer_1[j] = update_weights(weight_layer_1[j], delta_hidden_list[i][j], inputs[i])

    print("Error ", error)              #print error
    outputs.clear()