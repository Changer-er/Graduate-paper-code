<script setup>
import { waterGetAll,waterSearch } from '@/api/aticle.js';
 import { ref,onMounted } from 'vue'
 //调用ref,定义响应式数据
 const waterlist = ref([]);
 //声明函数
 const getAllWater =  async function(){
   let data = await waterGetAll();
   waterlist.value = data;
 }
 getAllWater();

 const searchConditions = ref({
    category:'',
    state:''
 })

 const search = async function() {
    let data = await waterSearch({...searchConditions.value});
    waterlist.data = data;
 }
 //声明钩子函数 onmounted
  onMounted(()=>{
    console.log('成功挂载');
  });
</script>

<template>

<button @click="increment">count:{{ count }}</button>
</template>