<script setup>
 import { ref,onMounted } from 'vue'
 //调用ref,定义响应式数据
 const count = ref(0);
 //声明函数
 function increment() {
    count.value++;
 }

 //声明钩子函数 onmounted
  onMounted(()=>{
    console.log('成功挂载');
  });
</script>

<template>
<button @click="increment">count:{{ count }}</button>
</template>