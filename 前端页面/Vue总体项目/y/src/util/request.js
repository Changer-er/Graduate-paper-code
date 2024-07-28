//定制请求示例
import axios from 'axios';

//获取所有水资源数据
const baseURL = "http://localhost:8080";
const instance = axios.create({baseURL})

//添加响应拦截器
instance.interceptors.response.use(
    result => {
        return result.data;
    },
    err => {
        alert('服务器异常');
        return Promise.reject(err);//异步的状态转成失败状态
    }
)
export default instance;