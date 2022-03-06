import Vue from 'vue'
import VueRouter from 'vue-router'
import LatentSpace from '../views/LatentSpace.vue'
import Compare from '../views/Compare.vue'
Vue.use(VueRouter)

const routes = [
  {
    path: '/',
    name: 'LatentSpace',
    component: LatentSpace
  },
  {
    path: '/vs',
    name: 'ObayashiKajima',
    component: Compare
  }

]

const router = new VueRouter({
  mode: 'history',
  base: process.env.BASE_URL,
  routes
})

export default router
