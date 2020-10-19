#include "ANN.h"
using namespace ANN;

namespace OCR{
	class Hangul_w{
	private:
		int		H_BLACK = 0;
		int		H_WHITE = -1;
		int		first = 19;
		int		middle = 21;
		int		last = 28;
		int		i_input = 256;
		int		d_input = 64;
		int		size = 16;
	public:
		char*	save_file = "Hangul_Neuron.txt";
		MLNN	hangul;
		Hangul_w(bool file_use = false){
			hangul.set_savefile(save_file);
			if (file_use)
				hangul.set_weight(save_file);
		}

		~Hangul_w(){};

		int Word_Search(double* input){
			Data_Convert(input, input, size);
			double*	an = hangul.get_answer(input);
			double	k = 0;
			int i = 0, f, m = 0, l = 0;
			for (i = 0; i < first; i++){
				if (an[i] < k) continue;
				k = an[i];
				f = i;
			}
			k = 0;
			for (; i < first + middle; i++){
				if (an[i] < k) continue;
				k = an[i];
				m = i;
			}
			k = 0;
			for (; i < first + middle + last; i++){
				if (an[i] < k) continue;
				k = an[i];
				l = i;
			}
			return (f * middle + m) * last + l + 0xAC00;
		}

		void Hangul_Learn(){
			char*	open_file = "NewHangul.txt";
			int		flr_cnt = 7;
			int		neu_cnt[] = { 64, 70, 40, 30, 10, 100, 68 };
			int		han_cnt = 11172;
			Data*	input = new Data[han_cnt];
			Data*	output = new Data[han_cnt];
			FILE*	stream = fopen(open_file, "r");
			Data_cnt(input, han_cnt, d_input);
			Data_cnt(output, han_cnt, neu_cnt[flr_cnt - 1]);
			int		f = 0;
			int		m = 0;
			int		l = 0;
			int		lb = 0;
			int*	ch = new int[han_cnt];
			for (int cnt = 0; cnt < han_cnt; cnt++)
				ch[cnt] = -1;
			for (int cnt = 0; cnt < han_cnt; cnt++) {
				int han = (f * middle + m) * last + l;
				ch[han] = cnt;
				output[cnt].arr[f] = 1;
				output[cnt].arr[first + m] = 1;
				output[cnt].arr[first + middle + l] = 1;
				f++;
				if (f == first) f = 0;
				m++;
				if (m == middle) m = 0;
				l++;
				if (l == last) l = 0;
				if (f == 0 && m == 0 && l == lb && cnt != han_cnt - 1){
					while (true){
						l++;
						if (l == last)l = 0;
						int h = (f*middle + m)*last + l;
						if (ch[h] == -1)break;
					}
					lb = l;
				}
			}
			for (int cnt = 0; cnt < han_cnt; cnt++){
				char str[5];
				double* temp = new double[i_input];
				for (int i = 0; i < i_input; i++)
					fscanf(stream, "%lf", &temp[i]);
				fscanf(stream, "%s\n", str);
				Data_Convert(temp, temp, size);
				for (int i = 0; i < d_input; i++)
					input[ch[cnt]].arr[i] = temp[i];
				delete[]temp;
			}
			hangul.set_weight(flr_cnt, neu_cnt);
			printf("Data setting complete\n");
			MLNN_FastLearn(hangul, han_cnt, input, output, 100);
			fclose(stream);
			delete[]input;
			delete[]output;
			delete[]ch;
		}
	};

	class Hangul_s{
	private:
		int		H_BLACK = 0;
		int		H_WHITE = -1;
		int		first = 19;
		int		middle = 21;
		int		last = 28;
		int		i_input = 256;
		int		d_input = 64;
		int		size = 16;
	public:
		char*	f_file = "First_Neuron.txt";
		char*	m_file = "Middle_Neuron.txt";
		char*	l_file = "Last_Neuron.txt";
		MLNN	f_MLNN;
		MLNN	m_MLNN;
		MLNN	l_MLNN;

		Hangul_s(bool file_use = false){
			f_MLNN.set_savefile(f_file);
			m_MLNN.set_savefile(m_file);
			l_MLNN.set_savefile(l_file);
			if (file_use){
				f_MLNN.set_weight(f_file);
				m_MLNN.set_weight(m_file);
				l_MLNN.set_weight(l_file);
			}
		}

		~Hangul_s(){};

		int Word_Search(double* input){
			Data_Convert(input, input, size);
			double*	an = f_MLNN.get_answer(input);
			double	k = 0;
			int i = 0, f = 0, m = 0, l = 0;
			for (i = 0; i < first; i++){
				if (an[i] < k) continue;
				k = an[i];
				f = i;
			}
			an = m_MLNN.get_answer(input);
			k = 0;
			for (i = 0; i < middle; i++){
				if (an[i] < k) continue;
				k = an[i];
				m = i;
			}
			an = l_MLNN.get_answer(input);
			k = 0;
			for (i = 0; i < last; i++){
				if (an[i] < k) continue;
				k = an[i];
				l = i;
			}
			return (f * middle + m) * last + l + 0xAC00;
		}

		void Hangul_Learn(){
			char*	open_file = "NewHangul.txt";
			int		f_flr = 4;
			int		m_flr = 4;
			int		l_flr = 4;
			int		f_neu[] = { 64, 38, 85, 19 };
			int		m_neu[] = { 64, 42, 105, 21 };
			int		l_neu[] = { 64, 56, 140, 28 };
			int		han_cnt = 11172;
			Data*	input = new Data[han_cnt];
			Data*	f_output = new Data[han_cnt];
			Data*	m_output = new Data[han_cnt];
			Data*	l_output = new Data[han_cnt];
			FILE*	stream = fopen(open_file, "r");
			Data_cnt(input, han_cnt, d_input);
			Data_cnt(f_output, han_cnt, first);
			Data_cnt(m_output, han_cnt, middle);
			Data_cnt(l_output, han_cnt, last);
			int		f = 0;
			int		m = 0;
			int		l = 0;
			int		lb = 0;
			int*	ch = new int[han_cnt];
			for (int cnt = 0; cnt < han_cnt; cnt++)
				ch[cnt] = -1;
			for (int cnt = 0; cnt < han_cnt; cnt++) {
				int han = (f * middle + m) * last + l;
				ch[han] = cnt;
				f_output[cnt].arr[f] = 1;
				m_output[cnt].arr[m] = 1;
				l_output[cnt].arr[l] = 1;
				f++;
				if (f == first) f = 0;
				m++;
				if (m == middle) m = 0;
				l++;
				if (l == last) l = 0;
				if (f == 0 && m == 0 && l == lb && cnt != han_cnt - 1){
					while (true){
						l++;
						if (l == last)l = 0;
						int h = (f*middle + m)*last + l;
						if (ch[h] == -1) break;
					}
					lb = l;
				}
			}
			for (int cnt = 0; cnt < han_cnt; cnt++){
				char str[5];
				double* temp = new double[i_input];
				for (int i = 0; i < i_input; i++)
					fscanf(stream, "%lf", &temp[i]);
				fscanf(stream, "%s\n", str);
				Data_Convert(temp, temp, size);
				for (int i = 0; i < d_input; i++)
					input[ch[cnt]].arr[i] = temp[i];
				delete[]temp;
			}
			fclose(stream);
			f_MLNN.set_weight(f_flr, f_neu);
			m_MLNN.set_weight(m_flr, m_neu);
			l_MLNN.set_weight(l_flr, l_neu);
			printf("Data setting complete\n");
			MLNN_FastLearn(f_MLNN, han_cnt, input, f_output, 111.72);
			MLNN_FastLearn(m_MLNN, han_cnt, input, m_output, 111.72);
			MLNN_FastLearn(l_MLNN, han_cnt, input, l_output, 111.72);
			delete[]input;
			delete[]f_output;
			delete[]m_output;
			delete[]l_output;
			delete[]ch;
		}
	};
}