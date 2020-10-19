#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <stdlib.h>
using namespace std;

//Artificial Neuron Network
namespace ANN{
	//Muit_Layer Neuron Network
	class MLNN{
	private:
		const char*	file;//���� ���� �̸�
		int			flr_cnt;//���� ����
		int*		neu_cnt;//�� ���� ��������
		double***	neu;//����ġ �迭
		double*		err;//���� ���� �迭
		double***	bfr_err;//���� ���� ���� �迭
		double*		output;//��°� ����迭
		double		err_sum;//���� ������ ��
		double		alpha;//�н���
		double		betha;//�н� ����ȭ
		void make_array(int _flr_cnt, int input[]){
			flr_cnt = _flr_cnt;
			neu_cnt = new int[flr_cnt];
			for (int i = 0; i < flr_cnt; i++)
				neu_cnt[i] = input[i];
			neu = new double**[flr_cnt - 1];//���������� �����
			for (int i = 0; i < flr_cnt - 1; i++){
				neu[i] = new double*[neu_cnt[i] + 1];//���� �����ϴ� ������ �� + �Ӱ谪 ����
				for (int j = 0; j < neu_cnt[i] + 1; j++)
					neu[i][j] = new double[neu_cnt[i + 1]];//����ġ n��
			}
			bfr_err = new double**[flr_cnt - 1];
			for (int i = 0; i < flr_cnt - 1; i++){
				bfr_err[i] = new double*[neu_cnt[i] + 1];
				for (int j = 0; j < neu_cnt[i] + 1; j++){
					bfr_err[i][j] = new double[neu_cnt[i + 1]];
					for (int k = 0; k < neu_cnt[i + 1]; k++)
						bfr_err[i][j][k] = 0;
				}
			}
		}
	public:
		MLNN(){
			alpha = 0.2;
			betha = 0.95;
			file = "neuron.txt";
		}

		~MLNN(){
			save_weight();
			for (int i = 0; i < flr_cnt - 1; i++){
				for (int j = 0; j <= neu_cnt[i]; j++)
					delete[] neu[i][j];
				delete[] neu[i];
			}
			delete[] neu;
		};

		void set_savefile(char _file[]){
			file = _file;
		}

		void set_weight(int flr, int input[]){
			make_array(flr, input);
			for (int i = 0; i < flr_cnt - 1; i++)
				for (int j = 0; j <= input[i]; j++)
					for (int k = 0; k < input[i + 1]; k++)
						neu[i][j][k] = ((double)rand() / (double)RAND_MAX - 0.5)*2.0;//-1.0~1.0 ������ �ʱⰪ
		}//����ġ ���� ����

		void set_weight(const char file[]){
			this->file = file;
			FILE* stream = fopen(this->file, "r");
			int flr;
			fscanf(stream, "%d", &flr);
			int *input;
			input = new int[flr];
			for (int i = 0; i < flr; i++)
				fscanf(stream, "%d", &input[i]);
			make_array(flr, input);
			for (int i = 0; i < flr_cnt - 1; i++)
				for (int j = 0; j <= neu_cnt[i]; j++)
					for (int k = 0; k < neu_cnt[i + 1]; k++)
						fscanf(stream, "%lf", &neu[i][j][k]);
			fscanf(stream, "%lf", &alpha);
			fclose(stream);
		}//����ġ txt�κ��� �Է�

		void save_weight(){
			FILE* stream = fopen(file, "w");
			if (stream == NULL)
				stream = fopen("neuron.txt", "w");
			fprintf(stream, "%d\n", flr_cnt);
			for (int i = 0; i < flr_cnt; i++)
				fprintf(stream, "%d ", neu_cnt[i]);
			fprintf(stream, "\n");
			for (int i = 0; i < flr_cnt - 1; i++)
				for (int j = 0; j <= neu_cnt[i]; j++)
					for (int k = 0; k < neu_cnt[i + 1]; k++)
						fprintf(stream, "%.20lf\n", neu[i][j][k]);
			fprintf(stream, "%.20lf\n", alpha);
			fclose(stream);
		}//����ġ txt�� ����

		double& at(int flr, int now, int next){
			return neu[flr][now][next];
		}

		double sigmoid(double x){
			return (1.0 / (1.0 + exp((-1) * x)));
		}

		void learn(double _input[], double _output[]) {
			output = new double[neu_cnt[flr_cnt - 1]];
			for (int i = 0; i < neu_cnt[flr_cnt - 1]; i++)
				output[i] = _output[i];
			err_sum = 0;//���� ������ �ʱ�ȭ
			calculate(_input);
			delete[]err;
			delete[]output;
		}//�н� �Լ�

		double* get_answer(double _input[]){
			calculate(_input, false);
			return output;
		}//�н������͸� ���Ͽ� �� ���

		void calculate(double _input[], bool learn = true, int next_flr = 1){
			if (next_flr == flr_cnt){//������ �������� ������̸�
				if (!learn){//�н��� ��ǥ�� �ƴϸ� ��� ���� ��� �迭�� �����Ѵ�
					output = new double[neu_cnt[flr_cnt - 1]];
					for (int i = 0; i < neu_cnt[flr_cnt - 1]; i++)
						output[i] = _input[i];
					return;
				}
				err = new double[neu_cnt[flr_cnt - 1]];
				for (int i = 0; i < neu_cnt[flr_cnt - 1]; i++)
					err[i] = output[i];
				for (int _out = 0; _out < neu_cnt[flr_cnt - 1]; _out++)
					err[_out] -= _input[_out];//����
				for (int _err = 0; _err < neu_cnt[flr_cnt - 1]; _err++)
					err_sum += err[_err] * err[_err];//���� ������ ���ϱ�
				for (int _out = 0; _out < neu_cnt[flr_cnt - 1]; _out++)
					err[_out] = err[_out] * _input[_out] * (1 - _input[_out]);//����� ��������
				return;
			}
			double *save = new double[neu_cnt[next_flr]];
			int now_flr = next_flr - 1;
			for (int next = 0; next < neu_cnt[next_flr]; next++){
				double sum = 0;
				for (int now = 0; now < neu_cnt[now_flr]; now++)
					sum += _input[now] * neu[now_flr][now][next];
				sum -= 1.0*neu[now_flr][neu_cnt[now_flr]][next];
				save[next] = sigmoid(sum);
			}
			calculate(save, learn, next_flr + 1);
			delete[] save;
			if (!learn) return;//�н��� ��ǥ�� �ƴϸ�
			double *t_err = new double[neu_cnt[now_flr]];//�������� �������� �迭
			for (int _now = 0; _now < neu_cnt[now_flr]; _now++){
				t_err[_now] = 0;
				for (int _next = 0; _next < neu_cnt[next_flr]; _next++){
					if (now_flr)//���� ���� �Է����� �ƴ� ���
						t_err[_now] += neu[now_flr][_now][_next] * err[_next];//�������� ���ϱ�-1
					double w_err = alpha*_input[_now] * err[_next];//����ġ ������ ���ϱ�
					neu[now_flr][_now][_next] += w_err + betha * bfr_err[now_flr][_now][_next];//����ġ ����
					bfr_err[now_flr][_now][_next] = w_err;
				}
				if (now_flr)
					t_err[_now] = t_err[_now] * _input[_now] * (1.0 - _input[_now]);//�������� ���ϱ�-2
			}
			for (int _thresh = 0; _thresh < neu_cnt[next_flr]; _thresh++){
				double thresh_err = alpha*(-1.0)*err[_thresh];
				neu[now_flr][neu_cnt[now_flr]][_thresh] += thresh_err + betha * bfr_err[now_flr][neu_cnt[now_flr]][_thresh];//�Ӱ谪 ����
				bfr_err[now_flr][neu_cnt[now_flr]][_thresh] = thresh_err;
			}
			delete[]err;
			err = new double[neu_cnt[now_flr]];
			for (int i = 0; i < neu_cnt[now_flr]; i++)
				err[i] = t_err[i];
			delete[] t_err;
		}

		double get_err_sum(){
			return err_sum;
		}

		double get_alpha(){
			return alpha;
		}

		void set_alpha(double _alpha){
			alpha = _alpha;
		}
	};

	class Data{
	public:
		double *arr;
		Data(){};
		~Data(){};
		void set_array(int _cnt){ 
			arr = new double[_cnt];
			for (int i = 0; i < _cnt; i++)
				arr[i] = 0;
		}
	};

	void Data_cnt(Data* k, int cnt, int d_cnt){
		for (int i = 0; i < cnt; i++)
			k[i].set_array(d_cnt);
	}

	double _min(double x, double y){
		return (x < y) ? x : y;
	}

	double _max(double x, double y){
		return (x > y) ? x : y;
	}

	void MLNN_Learn(MLNN &net, int ex_cnt, Data* ex_in, Data* ex_out, double end_learn){
		double err_sum = 0;
		for (int l = 0, t = 0;; l++){
			net.learn(ex_in[l].arr, ex_out[l].arr);
			err_sum += net.get_err_sum();
			if (l == ex_cnt - 1){
				t++;
				l = 0;
				printf("%d : %.20lf\n", t, err_sum);
				if (err_sum < end_learn)
					break;
				err_sum = 0;
			}
		}
		net.save_weight();
	}

	void MLNN_FastLearn(MLNN &net, int ex_cnt, Data* ex_in, Data* ex_out, double end_learn){
		FILE* stream = fopen("Error.txt", "a");
		double err_bfr = 0;
		double err_sum = 0;
		double rate = 1.04;
		double mtp = 1.05;
		double dvd = 0.7;
		double max_alpha = 2.5;
		double min_alpha = 0.00001;
		for (int l = 0, t = 0;; l++){
			net.learn(ex_in[l].arr, ex_out[l].arr);
			err_sum += net.get_err_sum();
			if (l == ex_cnt - 1){
				t++;
				l = 0;
				//fprintf(stream, "%d : %.20lf\n", t, err_sum);
				printf("%d : %.20lf\n", t, err_sum);
				if (err_sum < end_learn)
					break;
				double alpha = net.get_alpha();
				if (err_bfr > err_sum*rate)//err_sum�� ���� ���� �̻����� ���ϸ�
					alpha = _max(min_alpha, alpha*dvd);//alpha���� �����
				else//�׷��� ������
					alpha = _min(max_alpha, alpha*mtp);//alpha���� �ִ밪 �̳����� ������Ų��
				net.set_alpha(alpha);//alpha�� ����
				err_bfr = err_sum;
				err_sum = 0;
			}
		}
		fclose(stream);
		ex_in = NULL;
		ex_out = NULL;
		net.save_weight();
	}

	void Data_Convert(double* &input, double* &convert, int size){
		double* temp = new double[size * 4];
		double* save = new double[size*size];
		for (int i = 0; i < size*size; i++)
			save[i] = input[i];
		delete[]convert;
		int D_BLACK = 0;
		int p2 = size * 2;
		for (int i = 0; i < size; i++){
			int black = 0;
			int space = 0;
			double mtp = 3.0;
			bool TF = false;
			int p1 = i * 2;
			for (int j = 0; j < size; j++){
				int pos = i*size + j;
				if (save[pos] == D_BLACK){
					black++;
					TF = true;
				}
				else if (TF) {
					space++;
					TF = false;
				}
			}
			if (TF) space++;
			temp[p1] = (double)black / (double)size * mtp;
			temp[(p1 + 1)] = (double)space / (double)size * mtp;
			black = 0;
			space = 0;
			TF = false;
			for (int j = 0; j < size; j++){
				int pos = i + j*size;
				if (save[pos] == D_BLACK){
					black++;
					TF = true;
				}
				else if (TF) {
					space++;
					TF = false;
				}
			}
			if (TF) space++;
			temp[(p1 + p2)] = (double)black / (double)size * mtp;
			temp[(p1 + p2 + 1)] = (double)space / (double)size * mtp;
		}
		convert = new double[size * 4];
		for (int i = 0; i < size * 4; i++)
			convert[i] = temp[i];
		delete[]temp;
		delete[]save;
	}
}