import Vue from 'vue'
import VueRouter from 'vue-router'
import LatentSpace from '../views/LatentSpace.vue'
import Compare from '../views/Compare.vue'
import svgSample from '../views/svgSample.vue'
import activeLearning from '../views/activeLearning.vue'
import collaboration from '../views/collaboration.vue'
import prediction from '../components/prediction.vue'

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
  },
  {
    path: '/svg',
    name: 'svg',
    component: svgSample
  },
  {
    path: '/al',
    name: 'activeLearning',
    component: activeLearning
  },
  {
    path: '/collaboration',
    name: 'collaboration',
    component: collaboration
  },
  {
    path: '/prediction',
    name: 'prediction',
    component: prediction,
    props: true
  }

]

const router = new VueRouter({
  mode: 'history',
  base: process.env.BASE_URL,
  routes
})

export default router
