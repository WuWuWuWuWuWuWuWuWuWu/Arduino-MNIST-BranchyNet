#include <Arduino.h>
#include <avr/pgmspace.h>


#include "trunk_1_uint8.tflite.h"
#include "trunk_2_uint8.tflite.h"
#include "trunk_3_uint8.tflite.h"
#include "branch_1_uint8.tflite.h"
#include "branch_2_uint8.tflite.h"
#include "branch_3_uint8.tflite.h"
#include "branch_4_uint8.tflite.h"

#include "image_data.h"
#include "image_list.h"

#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"


// Globals, used for compatibility with Arduino-style sketches.
namespace {
	const tflite::Model* module1 = nullptr;
	const tflite::Model* module2 = nullptr;
	const tflite::Model* module3 = nullptr;
	const tflite::Model* module4 = nullptr;
	const tflite::Model* module5 = nullptr;
	const tflite::Model* module6 = nullptr;
	const tflite::Model* module7 = nullptr;

	tflite::MicroInterpreter* trunk_1 = nullptr;
	tflite::MicroInterpreter* branch_1 = nullptr;
	tflite::MicroInterpreter* trunk_2 = nullptr;
	tflite::MicroInterpreter* branch_2 = nullptr;
	tflite::MicroInterpreter* trunk_3 = nullptr;
	tflite::MicroInterpreter* branch_3 = nullptr;
	tflite::MicroInterpreter* branch_4 = nullptr;

	TfLiteTensor* input_1 = nullptr;
	TfLiteTensor* output_1 = nullptr;
	TfLiteTensor* input_2 = nullptr;
	TfLiteTensor* output_2 = nullptr;
	TfLiteTensor* input_3 = nullptr;
	TfLiteTensor* output_3 = nullptr;
	TfLiteTensor* input_4 = nullptr;
	TfLiteTensor* output_4 = nullptr;
	TfLiteTensor* input_5 = nullptr;
	TfLiteTensor* output_5 = nullptr;
	TfLiteTensor* input_6 = nullptr;
	TfLiteTensor* output_6 = nullptr;
	TfLiteTensor* input_7 = nullptr;
	TfLiteTensor* output_7 = nullptr;

	constexpr int kTensorArenaSize1 = 1024 * 8;
	constexpr int kTensorArenaSize2 = 1024 * 36;
	constexpr int kTensorArenaSize3 = 1024 * 8;
	constexpr int kTensorArenaSize4 = 1024 * 13;
	constexpr int kTensorArenaSize5 = 1024 * 33;
	constexpr int kTensorArenaSize6 = 1024 * 3;
	constexpr int kTensorArenaSize7 = 1024 * 14;

	uint8_t tensor_arena1[kTensorArenaSize1];
	uint8_t tensor_arena2[kTensorArenaSize2];
	uint8_t tensor_arena3[kTensorArenaSize3];
	uint8_t tensor_arena4[kTensorArenaSize4];
	uint8_t tensor_arena5[kTensorArenaSize5];
	uint8_t tensor_arena6[kTensorArenaSize6];
	uint8_t tensor_arena7[kTensorArenaSize7];
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
	tflite::InitializeTarget();

	// Map the model into a usable data structure. This doesn't involve any
	// copying or parsing, it's a very lightweight operation.
	module1 = tflite::GetModel(trunk_1_uint8_tflite);
	module2 = tflite::GetModel(branch_1_uint8_tflite);
	module3= tflite::GetModel(trunk_2_uint8_tflite);
	module4 = tflite::GetModel(branch_2_uint8_tflite);
	module5= tflite::GetModel(trunk_3_uint8_tflite);
	module6 = tflite::GetModel(branch_3_uint8_tflite);
	module7 = tflite::GetModel(branch_4_uint8_tflite);
	if(module1 ->version() != TFLITE_SCHEMA_VERSION) MicroPrintf("Model provided is schema version %d not equal to supported version %d.", module1 ->version(), TFLITE_SCHEMA_VERSION);
	if(module2 ->version() != TFLITE_SCHEMA_VERSION) MicroPrintf("Model provided is schema version %d not equal to supported version %d.", module2 ->version(), TFLITE_SCHEMA_VERSION);
	if(module3 ->version() != TFLITE_SCHEMA_VERSION) MicroPrintf("Model provided is schema version %d not equal to supported version %d.", module3 ->version(), TFLITE_SCHEMA_VERSION);
	if(module4 ->version() != TFLITE_SCHEMA_VERSION) MicroPrintf("Model provided is schema version %d not equal to supported version %d.", module4 ->version(), TFLITE_SCHEMA_VERSION);
	if(module5 ->version() != TFLITE_SCHEMA_VERSION) MicroPrintf("Model provided is schema version %d not equal to supported version %d.", module5 ->version(), TFLITE_SCHEMA_VERSION);
	if(module6 ->version() != TFLITE_SCHEMA_VERSION) MicroPrintf("Model provided is schema version %d not equal to supported version %d.", module6 ->version(), TFLITE_SCHEMA_VERSION);
	if(module7 ->version() != TFLITE_SCHEMA_VERSION) MicroPrintf("Model provided is schema version %d not equal to supported version %d.", module7 ->version(), TFLITE_SCHEMA_VERSION);


	// This pulls in all the operation implementations we need.
	// NOLINTNEXTLINE(runtime-global-variables)
	static tflite::MicroMutableOpResolver<5> resolver_1;
	resolver_1.AddQuantize();
	resolver_1.AddConv2D();
	resolver_1.AddDepthwiseConv2D();
	resolver_1.AddRelu();
	resolver_1.AddAveragePool2D();
	static tflite::MicroMutableOpResolver<5> resolver_2;
	resolver_2.AddQuantize();
	resolver_2.AddAveragePool2D();
	resolver_2.AddReshape();
	resolver_2.AddSoftmax();
	resolver_2.AddFullyConnected();
	// // // Build an interpreter to run the model with.
	static tflite::MicroInterpreter interpreter_1(module1 , resolver_1, tensor_arena1, kTensorArenaSize1);
	trunk_1 = &interpreter_1;
	static tflite::MicroInterpreter interpreter_2(module2 , resolver_2, tensor_arena2, kTensorArenaSize2);
	branch_1 = &interpreter_2;
	static tflite::MicroInterpreter interpreter_3(module3 , resolver_1, tensor_arena3, kTensorArenaSize3);
	trunk_2 = &interpreter_3;
	static tflite::MicroInterpreter interpreter_4(module4 , resolver_2, tensor_arena4, kTensorArenaSize4);
	branch_2 = &interpreter_4;
	static tflite::MicroInterpreter interpreter_5(module5 , resolver_2, tensor_arena5, kTensorArenaSize5);
	trunk_3 = &interpreter_5;
	static tflite::MicroInterpreter interpreter_6(module6 , resolver_2, tensor_arena6, kTensorArenaSize6);
	branch_3 = &interpreter_6;
	static tflite::MicroInterpreter interpreter_7(module7 , resolver_2, tensor_arena7, kTensorArenaSize7);
	branch_4 = &interpreter_7;

	// // Allocate memory from the tensor_arena for the model's tensors.
	TfLiteStatus allocate_status1 = trunk_1->AllocateTensors();
	TfLiteStatus allocate_status2 = branch_1->AllocateTensors();
	TfLiteStatus allocate_status3 = trunk_2->AllocateTensors();
	TfLiteStatus allocate_status4 = branch_2->AllocateTensors();
	TfLiteStatus allocate_status5 = trunk_3->AllocateTensors();
	TfLiteStatus allocate_status6 = branch_3->AllocateTensors();
	TfLiteStatus allocate_status7 = branch_4->AllocateTensors();

	if(allocate_status1 != kTfLiteOk) MicroPrintf("AllocateTensors1() failed");
	if(allocate_status2 != kTfLiteOk) MicroPrintf("AllocateTensors2() failed");
	if(allocate_status3 != kTfLiteOk) MicroPrintf("AllocateTensors3() failed");
	if(allocate_status4 != kTfLiteOk) MicroPrintf("AllocateTensors4() failed");
	if(allocate_status5 != kTfLiteOk) MicroPrintf("AllocateTensors5() failed");
	if(allocate_status6 != kTfLiteOk) MicroPrintf("AllocateTensors6() failed");
	if(allocate_status7 != kTfLiteOk) MicroPrintf("AllocateTensors7() failed");

	// // Obtain pointers to the model's input and output tensors.
	input_1 = trunk_1->input(0);
	output_1 = trunk_1->output(0);
	input_2 = branch_1->input(0);
	output_2 = branch_1->output(0);
	input_3 = trunk_2->input(0);
	output_3 = trunk_2->output(0);
	input_4 = branch_2->input(0);
	output_4 = branch_2->output(0);
	input_5 = trunk_3->input(0);
	output_5 = trunk_3->output(0);
	input_6 = branch_3->input(0);
	output_6 = branch_3->output(0);
	input_7 = branch_4->input(0);
	output_7 = branch_4->output(0);
}

void loop() {
	int correct = 0, exit = 3;
	float time = 0;
	float* accuracy_list = new float[image_count];
	float* latency_list = new float[image_count];


	Serial.println("index\t\tlabel\tpredict\taccuracy\tlatency");
	for(int i= 0; i < image_count; i++) {
		accuracy_list[i]= 0;
		latency_list[i]= 0;
		int predict = 0;
		uint8_t max = 0;


		float start_time = millis();
		// Place the quantized input in the model's input tensor
		for(int j= 0; j < 784; j++) input_1->data.uint8[j] = pgm_read_byte_near(&(image_data[i][j]));
		// Run inference, and report any error
		TfLiteStatus invoke_status = trunk_1->Invoke();
		if(invoke_status != kTfLiteOk) MicroPrintf("trunk_1 Invoke failed\n");


		if(exit == 1) {
			for(int j= 0; j < 3456; j++) input_2->data.uint8[j] = output_1 -> data.uint8[j];
			TfLiteStatus invoke_status = branch_1->Invoke();
			if(invoke_status != kTfLiteOk) MicroPrintf("branch_1 Invoke failed\n");

			max = 0;
			for(int j= 0; j < 10; j++) {
				if(output_2->data.uint8[j] > max) {
					predict = j;
					max = output_2->data.uint8[j]; 
				}
			}
		}

		else if(exit > 1) {
			for(int j= 0; j < 3456; j++) input_3->data.uint8[j] = output_1 -> data.uint8[j];
			TfLiteStatus invoke_status = trunk_2->Invoke();
			if(invoke_status != kTfLiteOk) MicroPrintf("trunk_2 Invoke failed\n");


			if(exit == 2) {
				for(int j= 0; j < 1024; j++) input_4->data.uint8[j] = output_3 -> data.uint8[j];
				TfLiteStatus invoke_status = branch_2->Invoke();
				if(invoke_status != kTfLiteOk) MicroPrintf("branch_2 Invoke failed\n");

				max = 0;
				for(int j= 0; j < 10; j++) {
					if(output_4->data.uint8[j] > max) {
						predict = j;
						max = output_4->data.uint8[j]; 
					}
				}
			}		
			else if(exit > 2) {
				for(int j= 0; j < 1024; j++) input_5->data.uint8[j] = output_3 -> data.uint8[j];
				TfLiteStatus invoke_status = trunk_3->Invoke();
				if(invoke_status != kTfLiteOk) MicroPrintf("trunk_3 Invoke failed\n");


				if(exit == 3) {
					for(int j= 0; j < 120; j++) input_6->data.uint8[j] = output_5 -> data.uint8[j];
					TfLiteStatus invoke_status = branch_3->Invoke();
					if(invoke_status != kTfLiteOk) MicroPrintf("branch_3 Invoke failed\n");

					max = 0;
					for(int j= 0; j < 10; j++) {
						if(output_6->data.uint8[j] > max) {
							predict = j;
							max = output_6->data.uint8[j]; 
						}
					}
				}


				else if(exit == 4) {
					for(int j= 0; j < 120; j++) input_7->data.uint8[j] = output_5 -> data.uint8[j];
					TfLiteStatus invoke_status = branch_4->Invoke();
					if(invoke_status != kTfLiteOk) MicroPrintf("branch_4 Invoke failed\n");

					max = 0;
					for(int j= 0; j < 10; j++) {
						if(output_7->data.uint8[j] > max) {
							predict = j;
							max = output_7->data.uint8[j]; 
						}
					} 
				}
			}
		}


		time += (millis() - start_time);
		if(predict == image_list[i]) correct += 1;
		Serial.println("\t" +String(i + 1) + "\t\t" + String(image_list[i]) + "\t\t" + String(predict) + "\t\t" + String(double(correct) / double(i + 1)) + "\t\t" + String(time / double(i + 1)));
		accuracy_list[i] = double(correct) / double(i + 1);
		latency_list[i] = time / double(i + 1);
		delay(100);
	}


	Serial.println("accuracy:\t" + String(double(correct ) / double(image_count)));
	Serial.print("\t[");
	for(int i= 0; i  < image_count; i++) {
		Serial.print(String(accuracy_list[i]));
		if(i != image_count - 1) Serial.print(", ");
	}
	Serial.println("]");


	Serial.println("latency:\t" + String(time / double(image_count)));
	Serial.print("\t[");
	for(int i= 0; i  < image_count; i++) {
		Serial.print(String(latency_list[i]));
		if(i != image_count - 1) Serial.print(", ");
	}
	Serial.println("]");
	delay(100000);
}