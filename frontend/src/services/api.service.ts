import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';

class ApiService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: '/api',
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 15000,
    });

    // JWT Auth Interceptor
    this.client.interceptors.request.use((config) => {
      const token = localStorage.getItem('token');
      if (token && config.headers) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    });

    // Global Response Handler
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          localStorage.removeItem('token');
          window.dispatchEvent(new Event('auth:unauthorized'));
        }
        return Promise.reject(error);
      }
    );
  }

  public async get<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const res: AxiosResponse<T> = await this.client.get(url, config);
    return res.data;
  }

  public async post<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const res: AxiosResponse<T> = await this.client.post(url, data, config);
    return res.data;
  }

  public async put<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> {
    const res: AxiosResponse<T> = await this.client.put(url, data, config);
    return res.data;
  }

  public async delete<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const res: AxiosResponse<T> = await this.client.delete(url, config);
    return res.data;
  }
}

export const apiService = new ApiService();
export default apiService;
