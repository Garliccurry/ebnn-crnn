#ifndef EBNN_H_INCLUDED
#define EBNN_H_INCLUDED
/* ———————————————————————————————————————— */
/*  ebnn网络底层                            */
/*  File    :ebnn.h                         */
/*  Author  :JiaLi.Ou   <109553196@qq.com>  */
/*  Others  :                               */
/*  Note    :The int8 quantization layer    */
/*    only passed operational testing, and  */
/*    inter layer testing was not conducted */
/* ———————————————————————————————————————— */
#include <stdio.h>
#include <stdint.h>
#include <float.h>
#include <limits.h>
#include <math.h>

#define MAX(a,b) ((a) > (b) ? a : b)
#define MIN(a,b) ((a) < (b) ? a : b)
#define CEIL_POS(X) ((X-(int)(X)) > 0 ? (int)(X+1) : (int)(X))
#define CLAMP_0_255(X) ((int)(X) > 255 ? 255 : ((int)(X) < 0 ? 0 : (int)(X)))
static const uint8_t bits[8] = {0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01};


/* 网络层 */
static void flinear_layer(const float* Input, const uint8_t* Weight, float* Output,
                          const float* Bias, const float* Gamma,
                          const float* Beta, const float* Mean,
                          const float* Std, const int batch_size,
						  const int i_dim,const int o_dim);
static void b8linear_layer(const uint8_t* Input, const uint8_t* Weight, uint8_t* Output,
                          const float* Bias, const float* Gamma,
                          const float* Beta, const float* Mean,
                          const float* Std, const int batch_size,
						  const int i_dim,const int o_dim);
static void flinear_sm_layer(const float* Input, const uint8_t* Weight, int* Output,
                             const float* Bias, const float* Gamma,
                             const float* Beta, const float* Mean,
                             const float* Std, const int batch_size,
                             const int i_dim,const int o_dim);
static void fflinear_sm_layer(const float* Input, const float* Weight, int* Output,
                             const float* Bias, const float* Gamma,
                             const float* Beta, const float* Mean,
                             const float* Std, const int batch_size,
                             const int i_dim,const int o_dim);
static void fconv_layer(const float* Input, const uint8_t* Weight, float* Output,
                        const float* Bias, const float* Gamma, const float* Beta,
                        const float* Mean, const float* Std, const int batch_size,
                        const int num_w, const int w, const int h, const int d,
                        const int kw, const int kh, const int sw, const int sh,
                        const int pw, const int ph, const int pl_w,
                        const int pl_h, const int pl_sw, const int pl_sh,
                        const int pl_pw, const int pl_ph);
static void b8conv_layer(const uint8_t* Input, const uint8_t* Weight, uint8_t* Output,
                        const float* Bias, const float* Gamma, const float* Beta,
                        const float* Mean, const float* Std, const int batch_size,
                        const int o_dim, const int w, const int h, const int d,
                        const int kw, const int kh, const int sw, const int sh,
                        const int pw, const int ph, const int pl_w,
                        const int pl_h, const int pl_sw, const int pl_sh,
                        const int pl_pw, const int pl_ph);
static void fBidLSTM_layer(const float* Input, const uint8_t* Weight_ih_f,
                        const uint8_t* Weight_ih_r, const uint8_t* Weight_hh_f,
                        const uint8_t* Weight_hh_r, float* Output, const float* Gamma,
                        const float* Beta, const float* Mean, const float* Std,
                        const int i_dim, const int h_dim, const int len_s);
static void permute021_layer(const float* Input, float* Output, const int pre_d,
                          const int pre_h, const int pre_w);
static void CTC_layer(const int* Input, int* output, const int len_s, const int len_o);

/* 网络层辅助函数 */
static void LSTM(const float* Input, const uint8_t* Weight_ih, const uint8_t* Weight_hh,
                 float* Output,const float* Gamma, const float* Beta,const float* Mean,
                 const float* Std, const float* h_pre, float* h_now, const float* c_pre,
                 float* c_now, const int i_dim, const int h_dim);
static void fconv(const float* Input, const uint8_t* Weight, float* Output,
                  const int o_idx, const float Bias, const float Gamma,
                  const float Beta, const float Mean, const float Std,
                  const int w, const int h, const int d, const int kw,
                  const int kh, const int sw, const int sh, const int pw,
                  const int ph, const int pl_w, const int pl_h, const int pl_sw,
                  const int pl_sh, const int pl_pw, const int pl_ph);
static float fdot_3d(const float* Input, const uint8_t* Weight, const int x, const int y,
                     const int w, const int h, const int d, const int kw,
                     const int kh);
static void b8conv(const uint8_t* Input, const uint8_t* Weight, uint8_t* Output,
                  const int o_idx, const float Bias, const float Gamma,
                  const float Beta, const float Mean, const float Std,
                  const int w, const int h, const int d, const int kw,
                  const int kh, const int sw, const int sh, const int pw,
                  const int ph, const int pl_w, const int pl_h, const int pl_sw,
                  const int pl_sh, const int pl_pw, const int pl_ph);
static float b8dot_3d(const uint8_t* Input, const uint8_t* Weight, const int x, const int y,
                     const int w, const int h, const int d, const int kw,
                     const int kh);
static float fdot(const float* Input, const uint8_t* Weight, const int input_num);
static int b8dot(const uint8_t* Input, const uint8_t* Weight, const int input_num);
static float qdot(const float* Input, const uint8_t* Weight, const int input_num);
static float ffdot(const float* Input, const float* Weight, const int input_num);
static float batch_norm(float f, const float Gamma, const float Beta,
                        const float Mean, const float Std);
static float sigmoid(float x);
static float Hard_sigmoid(float x);
static float Hard_tanh(float x);

/* 寻址函数 */
static int idx_2d(const int i, const int j, const int rows);
static int conv_idx(const int pl_i, const int x, const int kx, const int sx,
                    const int px);
static int convpool_size(const int x, const int kx, const int sx, const int px,
                         const int pl_x, const int pl_sx, const int pl_px);

/* 位运算函数 */
static int nthbitset_arr(const uint8_t* const arr, const int n);
static int popcnt8(const uint8_t v);


/* ———————————————————————————————————————— */
/* ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ 函数实例 ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ */
/* ———————————————————————————————————————— */


/* 网络层 */
static void flinear_layer(const float* Input, const uint8_t* Weight, float* Output,
                          const float* Bias, const float* Gamma,
                          const float* Beta, const float* Mean,
                          const float* Std, const int batch_size,
						  const int i_dim,const int o_dim)
{
	int i, j, int_i_num, o_idx, w_idx_bias;
	const float *i_re_idx;
	float res;

    int_i_num = (i_dim + 7)/8;

	/* 浮点8个bytes矩阵乘加运算 */
	for(i = 0 ; i < batch_size ; ++i){
	    o_idx = i * o_dim;
		i_re_idx = Input + (i * i_dim);
		for(j = 0 ; j < o_dim ; ++j){
			w_idx_bias = j * int_i_num;
			res = fdot(i_re_idx, Weight + w_idx_bias, i_dim);
			res += Bias[j];
			res = batch_norm(res, Gamma[j], Beta[j], Mean[j], Std[j]);
			/* 存储结果 */
			Output[j + o_idx] = res;
		}
	}
}

static void b8linear_layer(const uint8_t* Input, const uint8_t* Weight, uint8_t* Output,
                          const float* Bias, const float* Gamma,
                          const float* Beta, const float* Mean,
                          const float* Std, const int batch_size,
						  const int i_dim,const int o_dim)
{
	int i, j, int_i_num, o_idx, w_idx_bias, cl_val;
	const uint8_t *i_re_idx;
	float res;

    int_i_num = (i_dim + 7)/8;

	/* 浮点8个bytes矩阵乘加运算 */
	for(i = 0 ; i < batch_size ; ++i){
	    o_idx = i * o_dim;
		i_re_idx = Input + (i * i_dim);
		for(j = 0 ; j < o_dim ; ++j){
			w_idx_bias = j * int_i_num;
			res = b8dot(i_re_idx, Weight + w_idx_bias, i_dim)/255.0;

            printf("res1:%f\r\n",res);
			res += Bias[j];
            printf("res2:%f\r\n",res);
			res = batch_norm(res, Gamma[j], Beta[j], Mean[j], Std[j])*127.5 + 127.5;
			/* 存储结果 */

            printf("res3:%f\r\n",res);
			Output[j + o_idx] = CLAMP_0_255(res);

            printf("op:%d\r\n",Output[j + o_idx]);

		}
	}
}

static void flinear_sm_layer(const float* Input, const uint8_t* Weight, int* Output,
                             const float* Bias, const float* Gamma,
                             const float* Beta, const float* Mean,
                             const float* Std, const int batch_size,
                             const int i_dim,const int o_dim)
{
	int i, j, max_idx, w_idx_bias;
	const float *i_re_idx;
	float max_res, res;

	/* 浮点8个bytes矩阵乘加运算 */
	for(i = 0 ; i < batch_size ; ++i){
		i_re_idx = Input + i * i_dim;
		max_res = -FLT_MAX;
		for(j = 0 ; j < o_dim ; ++j){
			w_idx_bias = j * i_dim;
			res = qdot(i_re_idx, Weight + w_idx_bias, i_dim);
			res += Bias[j];
			res = batch_norm(res, Gamma[j], Beta[j], Mean[j], Std[j]);
			/* 比较结果 */
			if (res > max_res) {
				max_idx = j;
				max_res = res;
			}
			/* 存储结果 */
            Output[i] = max_idx;
		}
	}
}

static void fflinear_sm_layer(const float* Input, const float* Weight, int* Output,
                             const float* Bias, const float* Gamma,
                             const float* Beta, const float* Mean,
                             const float* Std, const int batch_size,
                             const int i_dim,const int o_dim)
{
	int i, j, max_idx, w_idx_bias;
	const float *i_re_idx;
	float max_res, res;

	/* 浮点8个bytes矩阵乘加运算 */
	for(i = 0 ; i < batch_size ; ++i){
		i_re_idx = Input + i * i_dim;
		max_res = -FLT_MAX;
		for(j = 0 ; j < o_dim ; ++j){
			w_idx_bias = j * i_dim;
			res = ffdot(i_re_idx, Weight + w_idx_bias, i_dim);
			res += Bias[j];
			res = batch_norm(res, Gamma[j], Beta[j], Mean[j], Std[j]);
			/* 比较结果 */
			if (res > max_res) {
				max_idx = j;
				max_res = res;
			}
			/* 存储结果 */
            Output[i] = max_idx;
		}
	}
}

static void fconv_layer(const float* Input, const uint8_t* Weight, float* Output,
                        const float* Bias, const float* Gamma, const float* Beta,
                        const float* Mean, const float* Std, const int batch_size,
                        const int o_dim, const int w, const int h, const int d,
                        const int kw, const int kh, const int sw, const int sh,
                        const int pw, const int ph, const int pl_w,
                        const int pl_h, const int pl_sw, const int pl_sh,
                        const int pl_pw, const int pl_ph)
{
    int i, j, max_m, res_size, res_w, res_h, o_idx, i_idx, w_idx, whd, k_whd_bytes;

    o_idx = 0;
    res_w = convpool_size(w, kw, sw, pw, pl_w, pl_sw, pl_pw);
    res_h = convpool_size(h, kh, sh, ph, pl_h, pl_sh, pl_ph);
    res_size = res_w * res_h;
    max_m = res_size * batch_size * o_dim;
    whd = w*h*d;
    k_whd_bytes = CEIL_POS(kw*kh*d/8.0);

    /* 分批次、输出维度处理 */
    for (i = 0; i < batch_size; ++i) {
        for (j = 0; j < o_dim; ++j) {
            i_idx = i*whd;
            w_idx = j*k_whd_bytes;
            /* 卷积运算 */
            fconv(Input + i_idx, Weight + w_idx, Output, o_idx, Bias[j], Gamma[j],
                  Beta[j], Mean[j], Std[j], w, h, d, kw, kh, sw, sh, pw, ph,
                  pl_w, pl_h, pl_sw, pl_sh, pl_pw, pl_ph);
            o_idx += res_size;
        }
    }
}

static void b8conv_layer(const uint8_t* Input, const uint8_t* Weight, uint8_t* Output,
                        const float* Bias, const float* Gamma, const float* Beta,
                        const float* Mean, const float* Std, const int batch_size,
                        const int o_dim, const int w, const int h, const int d,
                        const int kw, const int kh, const int sw, const int sh,
                        const int pw, const int ph, const int pl_w,
                        const int pl_h, const int pl_sw, const int pl_sh,
                        const int pl_pw, const int pl_ph)
{
    int i, j, max_m, res_size, res_w, res_h, o_idx, i_idx, w_idx, whd, k_whd_bytes;

    o_idx = 0;
    res_w = convpool_size(w, kw, sw, pw, pl_w, pl_sw, pl_pw);
    res_h = convpool_size(h, kh, sh, ph, pl_h, pl_sh, pl_ph);
    res_size = res_w * res_h;
    max_m = res_size * batch_size * o_dim;
    whd = w*h*d;
    k_whd_bytes = CEIL_POS(kw*kh*d/8.0);

    /* 分批次、输出维度处理 */
    for (i = 0; i < batch_size; ++i) {
        for (j = 0; j < o_dim; ++j) {
            i_idx = i*whd;
            w_idx = j*k_whd_bytes;
            /* 卷积运算 */
            b8conv(Input + i_idx, Weight + w_idx, Output, o_idx, Bias[j], Gamma[j],
                  Beta[j], Mean[j], Std[j], w, h, d, kw, kh, sw, sh, pw, ph,
                  pl_w, pl_h, pl_sw, pl_sh, pl_pw, pl_ph);
            o_idx += res_size;
        }
    }
}

static void fBidLSTM_layer(const float* Input, const uint8_t* Weight_ih_f,
                        const uint8_t* Weight_ih_r, const uint8_t* Weight_hh_f,
                        const uint8_t* Weight_hh_r, float* Output, const float* Gamma,
                        const float* Beta, const float* Mean, const float* Std,
                        const int i_dim, const int h_dim, const int len_s)
{
    int i, o_dim, i_idx, o_idx;
    o_dim = 2 * h_dim;

    /* 初始化 state */
    float h_state_0[h_dim], h_state_1[h_dim], c_state_0[h_dim], c_state_1[h_dim];
    for (int i = 0; i < h_dim; i++) {
        h_state_0[i] = 0;
        h_state_1[i] = 0;
        c_state_0[i] = 0;
        c_state_1[i] = 0;
    }
    /* 正向rnn */
    for (i = 0; i < len_s; ++i){
        i_idx = i * i_dim;
        o_idx = i * o_dim;
        if(i % 2 == 0){
            LSTM(Input + i_idx, Weight_ih_f, Weight_hh_f, Output + o_idx, Gamma, Beta,
                 Mean, Std, h_state_1, h_state_0, c_state_1, c_state_0, i_dim, h_dim);
        }
        else{
            LSTM(Input + i_idx, Weight_ih_f, Weight_hh_f, Output + o_idx, Gamma, Beta,
                 Mean, Std, h_state_0, h_state_1, c_state_0, c_state_1, i_dim, h_dim);
        }
    }
    /* 反向rnn */
    for (i = len_s-1; i >= 0; --i){
        i_idx = i * i_dim;
        o_idx = i * o_dim + h_dim;
        if(i % 2 == 0){
            LSTM(Input + i_idx, Weight_ih_r, Weight_hh_r, Output + o_idx, Gamma + h_dim, Beta + h_dim,
                 Mean + h_dim, Std + h_dim, h_state_1, h_state_0, c_state_1, c_state_0, i_dim, h_dim);
        }
        else{
            LSTM(Input + i_idx, Weight_ih_r, Weight_hh_r, Output + o_idx, Gamma + h_dim, Beta + h_dim,
                 Mean + h_dim, Std + h_dim, h_state_0, h_state_1, c_state_0, c_state_1, i_dim, h_dim);
        }
    }
}

static void permute021_layer(const float* Input, float* Output, const int pre_d,
                          const int pre_h, const int pre_w)
{
    int i, j, k, d_idx;

    /* [x][y][z] → [x][z][y] */
    for (i = 0; i < pre_d; ++i) {
        d_idx = i * pre_w * pre_h;
        for (j = 0; j < pre_h; ++j) {
            for (k = 0; k < pre_w; ++k) {
                Output[d_idx + k * pre_h + j] = Input[d_idx + j * pre_w + k];
            }
        }
    }

}

static void CTC_layer(const int* Input, int* output, const int len_s, const int len_o)
{
    int i, j;
    j = 0;

    for (i = 0; i<len_o; ++i){
        output[i] = 0;
    }

    for (i = 0; i<len_s; ++i){
        /* 处理非0值 */
        if(Input[i] != 0){
            if(j == 0){
                output[j] = Input[i];
                j++;
            }
            else if(output[j-1] != Input[i]){
                output[j] = Input[i];
                j++;
            }
        }
    }
}

/* 网络层辅助函数 */
static void LSTM(const float* Input, const uint8_t* Weight_ih, const uint8_t* Weight_hh,
                 float* Output,const float* Gamma, const float* Beta,const float* Mean,
                 const float* Std, const float* h_pre, float* h_now, const float* c_pre,
                 float* c_now, const int i_dim, const int h_dim)
{
    /*
    Wii = w_ih[       0      :  h_dim*i_dim ]   all/8bit
    Wif = w_ih[ h_dim*i_dim  : 2*h_dim*i_dim]   all/8bit
    Wio = w_ih[2*h_dim*i_dim : 3*h_dim*i_dim]   all/8bit
    Wic = w_ih[3*h_dim*i_dim : 4*h_dim*i_dim]   all/8bit

    Whi = w_hh[       0      :  h_dim*h_dim ]   all/8bit
    Whf = w_hh[ h_dim*i_dim  : 2*h_dim*h_dim]   all/8bit
    Who = w_hh[2*h_dim*i_dim : 3*h_dim*h_dim]   all/8bit
    Whc = w_hh[3*h_dim*i_dim : 4*h_dim*h_dim]   all/8bit
     */
    int i, j, k, input_bytes, hidden_bytes, size_ih_bytes, size_hh_bytes, w_idx, W_bit;
    float ip[h_dim], fg[h_dim], op[h_dim], cl[h_dim];

    input_bytes = CEIL_POS(i_dim/8.0);
    hidden_bytes = CEIL_POS(h_dim/8.0);
    size_ih_bytes = CEIL_POS(i_dim * h_dim/8.0);
    size_hh_bytes = CEIL_POS(h_dim * h_dim/8.0);

    /* 遗忘门 */
    for (i = 0; i < h_dim; ++i){
        fg[i] = 0;
        for (j = 0; j < input_bytes; ++j){
            w_idx = size_ih_bytes + i * input_bytes + j;
            for  (k = 0; k < 8; ++k){
                W_bit = (Weight_ih[w_idx] >> (7-k)) & 1;
                fg[i] += ((W_bit>0) ? Input[j*8+k] : -Input[j*8+k]);
                if(i == 0){
                    printf("w:%d I:%f fg:%f\r\n", W_bit, Input[j*8+k], fg[i]);
                }
            }
        }
        for (j = 0; j < hidden_bytes; ++j){
            w_idx = size_hh_bytes + i * hidden_bytes + j;
            for  (k = 0; k < 8; ++k){
                W_bit = (Weight_hh[w_idx] >> (7-k)) & 1;
                fg[i] += ((W_bit>0) ? h_pre[j*8+k] : -h_pre[j*8+k]);
            }
        }
        if(i == 0){
            printf("fg:%f\r\n", fg[i]);
        }
        fg[i] = Hard_sigmoid(fg[i]);
        if(i == 0){
            printf("fg:%f\r\n", fg[i]);
        }
    }

    /* 输入门 */
    for (i = 0; i < h_dim; ++i){
        ip[i] = 0;
        for (j = 0; j < input_bytes; ++j){
            w_idx = i * input_bytes + j;
            for  (k = 0; k < 8; ++k){
                W_bit = (Weight_ih[w_idx] >> (7-k)) & 1;
                ip[i] += ((W_bit>0) ? Input[j*8+k] : -Input[j*8+k]);
            }
        }
        for (j = 0; j < hidden_bytes; ++j){
            w_idx = i * hidden_bytes + j;
            for  (k = 0; k < 8; ++k){
                W_bit = (Weight_hh[w_idx] >> (7-k)) & 1;
                ip[i] += ((W_bit>0) ? h_pre[j*8+k] : -h_pre[j*8+k]);
            }
        }
        ip[i] = Hard_sigmoid(ip[i]);
        if(i == 0){
            printf("ip:%f\r\n", ip[i]);
        }
    }

    /* 候选门 */
    for (i = 0; i < h_dim; ++i){
        cl[i] = 0;
        for (j = 0; j < input_bytes; ++j){
            w_idx = 2 * size_ih_bytes + i * input_bytes + j;
            for  (k = 0; k < 8; ++k){
                W_bit = (Weight_ih[w_idx] >> (7-k)) & 1;
                cl[i] += ((W_bit>0) ? Input[j*8+k] : -Input[j*8+k]);
            }
        }
        for (j = 0; j < hidden_bytes; ++j){
            w_idx = 2 * size_hh_bytes + i * hidden_bytes + j;
            for  (k = 0; k < 8; ++k){
                W_bit = (Weight_hh[w_idx] >> (7-k)) & 1;
                cl[i] += ((W_bit>0) ? h_pre[j*8+k] : -h_pre[j*8+k]);
            }
        }
        cl[i] = Hard_tanh(cl[i]);
        if(i == 0){
            printf("cl:%f\r\n", cl[i]);
        }
    }

    /* 输出门 */
    for (i = 0; i < h_dim; ++i){
        op[i] = 0;
        for (j = 0; j < input_bytes; ++j){
            w_idx = 3 * size_ih_bytes + i * input_bytes + j;
            for  (k = 0; k < 8; ++k){
                W_bit = (Weight_ih[w_idx] >> (7-k)) & 1;
                op[i] += ((W_bit>0) ? Input[j*8+k] : -Input[j*8+k]);
            }
        }
        for (j = 0; j < hidden_bytes; ++j){
            w_idx = 3 * size_hh_bytes + i * hidden_bytes + j;
            for  (k = 0; k < 8; ++k){
                W_bit = (Weight_hh[w_idx] >> (7-k)) & 1;
                op[i] += ((W_bit>0) ? h_pre[j*8+k] : -h_pre[j*8+k]);
            }
        }
        op[i] = Hard_sigmoid(op[i]);
        if(i == 0){
                printf("op:%f\r\n", op[0]);
        }
    }

    /* 更新隐藏状态以及细胞状态 */
    for (int i = 0; i < h_dim; i++) {
        c_now[i] = fg[i] * c_pre[i] + ip[i] * cl[i];
        h_now[i] = op[i] * Hard_tanh(c_now[i]);
        Output[i] = batch_norm(h_now[i], Gamma[i], Beta[i], Mean[i], Std[i]);
        if(i == 0){
            printf("fg:%f  ip:%f  cl:%f  op:%f\r\n", fg[i], ip[i], cl[i], op[i]);
            printf("c_p:%f  h_p:%f  c_n:%f  h_n:%f  op:%f\r\n", c_pre[i], h_pre[i], c_now[i], h_now[i], Output[i]);
        }
    }

}


static void fconv(const float* Input, const uint8_t* Weight, float* Output,
                  const int o_idx, const float Bias, const float Gamma,
                  const float Beta, const float Mean, const float Std,
                  const int w, const int h, const int d, const int kw,
                  const int kh, const int sw, const int sh, const int pw,
                  const int ph, const int pl_w, const int pl_h, const int pl_sw,
                  const int pl_sh, const int pl_pw, const int pl_ph)
{
    int o_index, pl_i, pl_j, i, j, i_in, j_in, pl_i_max, pl_j_max, pl_w_start, pl_h_start;
    float res, max_res;

    /* 初始化参数 */
    o_index = o_idx;
    pl_i_max = (w - kw + 2*pw)/sw + (2*pl_pw) + 1;
    pl_j_max = (h - kh + 2*ph)/sh + (2*pl_ph) + 1;
    pl_w_start = pl_w + pl_pw - 1;
    pl_h_start = pl_h + pl_ph - 1;

    /* 历遍卷积-池化区域 */
    for (pl_j = -pl_ph; pl_j + pl_h_start < pl_j_max; pl_j += pl_sh) {
        for (pl_i = -pl_pw; pl_i + pl_w_start < pl_i_max; pl_i += pl_sw) {
            max_res = res = -FLT_MAX;
            int pl_j_pl_h = pl_j + pl_h;
            for (j_in = pl_j; j_in < pl_j_pl_h; ++j_in) {
                j = conv_idx(j_in, h, kh, sh, ph);
                int pl_i_pl_w = pl_i + pl_w;
                for (i_in = pl_i; i_in < pl_i_pl_w; ++i_in){
                    i = conv_idx(i_in, w, kw, sw, pw);
                    if (i >= -pw && j >= -ph) {
                        res = fdot_3d(Input, Weight, i, j, w, h, d, kw, kh);
                        max_res = MAX(res, max_res);
                    }
                }
            }
            max_res += Bias;
            max_res = batch_norm(max_res, Gamma, Beta, Mean, Std);

            /* 存储结果 */
            Output[o_index] = max_res;
            o_index++;
        }
    }
}


static float fdot_3d(const float* Input, const uint8_t* Weight, const int x, const int y,
                     const int w, const int h, const int d, const int kw,
                     const int kh)
{
    uint8_t  bit_set;
    int i, j, k, b_idx, I_d_bias, x_kw, y_kh;
    float a, res;
    const float *Input_slice;

    I_d_bias = w*h;
    res = 0;
    b_idx = 0;
    x_kw = x + kw;
    y_kh = y + kh;

    for (i = 0; i < d; ++i) {
        Input_slice = Input + I_d_bias*i;
        for (j = y; j < y_kh; ++j) {
            for (k = x; k < x_kw; ++k) {
                /* 处理padding */
                if (j < 0 || j > h-1 || k < 0 || k > w-1) {
                    a = 0.0;
                }
                else {
                    a = Input_slice[idx_2d(j, k, w)];
                }
                bit_set = nthbitset_arr(Weight, b_idx);
                res += bit_set ? a : -a;
                b_idx++;
            }
        }
    }
    return res;
}

static void b8conv(const uint8_t* Input, const uint8_t* Weight, uint8_t* Output,
                  const int o_idx, const float Bias, const float Gamma,
                  const float Beta, const float Mean, const float Std,
                  const int w, const int h, const int d, const int kw,
                  const int kh, const int sw, const int sh, const int pw,
                  const int ph, const int pl_w, const int pl_h, const int pl_sw,
                  const int pl_sh, const int pl_pw, const int pl_ph)
{
    int o_index, pl_i, pl_j, i, j, i_in, j_in, pl_i_max, pl_j_max, pl_w_start, pl_h_start;
    float res, max_res;

    /* 初始化参数 */
    o_index = o_idx;
    pl_i_max = (w - kw + 2*pw)/sw + (2*pl_pw) + 1;
    pl_j_max = (h - kh + 2*ph)/sh + (2*pl_ph) + 1;
    pl_w_start = pl_w + pl_pw - 1;
    pl_h_start = pl_h + pl_ph - 1;

    /* 历遍卷积-池化区域 */
    for (pl_j = -pl_ph; pl_j + pl_h_start < pl_j_max; pl_j += pl_sh) {
        for (pl_i = -pl_pw; pl_i + pl_w_start < pl_i_max; pl_i += pl_sw) {
            max_res = res = -FLT_MAX;
            int pl_j_pl_h = pl_j + pl_h;
            for (j_in = pl_j; j_in < pl_j_pl_h; ++j_in) {
                j = conv_idx(j_in, h, kh, sh, ph);
                int pl_i_pl_w = pl_i + pl_w;
                for (i_in = pl_i; i_in < pl_i_pl_w; ++i_in){
                    i = conv_idx(i_in, w, kw, sw, pw);
                    if (i >= -pw && j >= -ph) {
                        res = b8dot_3d(Input, Weight, i, j, w, h, d, kw, kh);
                        max_res = MAX(res, max_res);
                    }
                }
            }
//            max_res = max_res / 255.0;
            printf("res1:%f", max_res);
            max_res += Bias;
            printf("res2:%f", max_res);
            max_res = batch_norm(max_res, Gamma, Beta, Mean, Std);
            printf("res3:%f", max_res);
            max_res = max_res * 127.5 + 127.5;
            printf("res4:%f\r\n", max_res);

            /* 存储结果 */
            Output[o_index] = CLAMP_0_255(max_res);
            o_index++;
        }
    }
}


static float b8dot_3d(const uint8_t* Input, const uint8_t* Weight, const int x, const int y,
                     const int w, const int h, const int d, const int kw,
                     const int kh)
{
    uint8_t  bit_set, W_bit;
    int i, j, k, b_idx, I_d_bias, x_kw, y_kh;
    const uint8_t *Input_slice;
    float a, res;

    I_d_bias = w*h;
    res = 0;
    b_idx = 0;
    x_kw = x + kw;
    y_kh = y + kh;

    for (i = 0; i < d; ++i) {
        Input_slice = Input + I_d_bias*i;
        for (j = y; j < y_kh; ++j) {
            for (k = x; k < x_kw; ++k) {
                /* 处理padding */
                if (j < 0 || j > h-1 || k < 0 || k > w-1) {
                    a = 0.0;
                }
                else {
                    a = (Input_slice[idx_2d(j, k, w)] - 127.5)/127.5;
                }
                bit_set = nthbitset_arr(Weight, b_idx);
                res += bit_set ? a : -a;
                b_idx++;
            }
        }
    }
    return (int)res;
}


static float fdot(const float* Input, const uint8_t* Weight, const int input_num)
{
	int i, j, input_bytes, W_bit;
	double res = 0;

	input_bytes = CEIL_POS(input_num/8.0);

	for(i = 0; i < input_bytes; ++i){
	    for(j = 0; j < 8; ++j){
            W_bit = (Weight[i] >> (7-j)) & 1;
//            res += Input[i * 8 + j] * W_bit;
            res += ((W_bit>0) ? Input[i*8+j] : -Input[i*8+j]);
	    }
	}
	return res;
}

static int b8dot(const uint8_t* Input, const uint8_t* Weight, const int input_num)
{
    int i, j, input_bytes, res;
    uint8_t W_bit;
    input_bytes = CEIL_POS(input_num/8.0);
    res = 0;
    for (i = 0; i < input_bytes; ++i) {
        for(j = 0; j < 8; ++j){
            W_bit = -((Weight[i] >> (7-j)) & 1);    // -0→00000000 / -1→11111111
            res += (uint8_t)~(Input[i*8+j]^W_bit);
//            printf("i:%d W_bit:%d Input:%d 1c:%d res:%d\r\n",i ,W_bit, Input[i*8+j],(uint8_t)~(Input[i*8+j]^W_bit), res);
        }
    }
    res = (res*2 - input_num*255);
//    printf("allres:%d\r\n",res);
    return res;
}

static float qdot(const float* Input, const uint8_t* Weight, const int input_num)
{
    uint8_t num, S_val, M_val;
	int i, j, E_val;
	float W, res = 0;

	/* 实数解压为浮点数 */
	for(i = 0; i < input_num; ++i){
	    W = 0;
        S_val = (Weight[i] >> 7) & 1;
        num = 0;
        for(j = 4; j > 0; --j){
            num += ((Weight[i] >> (2+j)) & 1) << (j-1);
        }
        E_val = (int)num;
        M_val = (Weight[i] & 7) | 8;
        for(j = 1; j <= 4; ++j){
            W += pow(2,-E_val-j)*((M_val>>(4-j))&1);
        }
        W = (S_val > 0)? -W: W;
        res += W *Input[i];
	}
	return res;
}

static float ffdot(const float* Input, const float* Weight, const int input_num)
{
	int i;
	float  res = 0;

	/* 实数解压为浮点数 */
	for(i = 0; i < input_num; ++i){
        res += Weight[i] *Input[i];
	}
	return res;
}

static float batch_norm(float f, const float Gamma, const float Beta,
                        const float Mean, const float Std)
{
  f -= Mean;
  f /= Std;
  f *= Gamma;
  f += Beta;
  return f;
}

static float Hard_sigmoid(float x)
{
    if(x > 3) return 1;
    else if(x < -3) return 0;
    else return x/6+0.5;
}

static float Hard_tanh(float x)
{
    if(x > 1) return 1;
    else if(x < -1) return -1;
    else return x;
}


/* 寻址函数 */
static int idx_2d(const int i, const int j, const int rows)
{
  return i * rows + j;
}

static int conv_idx(const int pl_i, const int x, const int kx, const int sx,
                    const int px)
{
  int conv_sz = (x - kx + 2*px)/sx;
  return (pl_i < 0 || pl_i > conv_sz) ? -INT_MAX : pl_i * sx - px;
}

static int convpool_size(const int x, const int kx, const int sx, const int px,
                         const int pl_x, const int pl_sx, const int pl_px)
{
  return ((x - kx + 2*px)/sx - pl_x + (2*pl_px) + 1)/pl_sx + 1;
}


/* 位操作函数 */
static int nthbitset_arr(const uint8_t* const arr, const int n)
{
  return arr[n/8] & bits[n%8] ? 1 : 0;
}

static int popcnt8(const uint8_t v)
{
  uint8_t c;
  c = v - ((v >> 1) & 0x55);
  c = ((c >> 2) & 0x33) + (c & 0x33);
  return ((c >> 4) + c) & 0x0F;
}



#endif // EBNN_H_INCLUDED
