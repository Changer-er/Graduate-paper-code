/* import axios from 'axios';

//获取所有水资源数据
const baseURL = "http://localhost:8080";
const instance =axios.create({baseURL}) */

import reuqest from '@/util/request.js'

export function waterGetAll() {
    return reuqest.get("/water/getAll");
}
//搜索水资源数据
export function waterSearch(Conditions) {
    return reuqest.get("/water/getAll", { params: Conditions });
}