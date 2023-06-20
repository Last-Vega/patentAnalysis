<template>
  <canvas ref="canvasRef" class="fullscreen"></canvas>
</template>

<script>
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'
import { TransformControls } from 'three/examples/jsm/controls/TransformControls.js'
// import { TrackballControls } from 'three/examples/jsm/controls/TrackballControls.js'
import company from '../assets/company_0908_z-1.json'
import term from '../assets/term_0908_z-1.json'

export default {
  data () {
    /**
       * non-reactiveなデータ
       * * https://stackoverflow.com/a/54907413
       */
    this.scene = null
    this.camera = null
    this.renderer = null
    this.cube = null
    this.controls = null

    /**
       * reactiveなデータ
       * * ここでThree.jsのデータを定義すると、重くなるか動かなくなるので注意
       *   * https://stackoverflow.com/a/65732553
       */
    return {}
  },

  mounted () {
    this.init()
    this.addSprite()
    this.animate()
  },

  methods: {
    init () {
      this.scene = new THREE.Scene()
      this.camera = new THREE.PerspectiveCamera(
        75,
        window.innerWidth / window.innerHeight,
        0.1,
        1000
      )
      // this.renderer = new THREE.WebGLRenderer({ canvas: this.$refs.canvasRef })
      // this.renderer.clearColor()
      // this.renderer.setSize(window.innerWidth, window.innerHeight)
      // this.renderer.setClearColor(new THREE.Color('black'))
      // this.document.body.appendChild(this.renderer.domElement)
      this.renderer = new THREE.WebGLRenderer()
      this.renderer.clearColor()
      this.renderer.setSize(window.innerWidth, window.innerHeight)
      this.renderer.setClearColor(new THREE.Color('white'))
      document.body.appendChild(this.renderer.domElement)

      // レンダリング解像度
      this.renderer.setSize(
        window.innerWidth,
        window.innerHeight,
        false
      )

      // this.camera.position.z = 5
      this.controls = new OrbitControls(this.camera, this.renderer.domElement)
      // this.controls = new TrackballControls(this.camera)
      const light = new THREE.DirectionalLight(0xffffff, 1)
      const lightHelper = new THREE.DirectionalLightHelper(light, 15)
      light.position.set(0.7, 0.7, 1)
      this.scene.add(light)
      this.scene.add(lightHelper)

      const axexHelper = new THREE.AxesHelper(1000)
      this.scene.add(axexHelper)

      const gridHelper = new THREE.GridHelper(1000, 100)
      this.scene.add(gridHelper)

      this.camera.position.set(100, 100, 100)
      this.controls.update()
    },

    addSprite () {
      const createSprite = (texture, scale, position) => {
        const spriteMaterial = new THREE.SpriteMaterial({ map: texture })
        const sprite = new THREE.Sprite(spriteMaterial)
        sprite.scale.set(scale.x, scale.y, scale.z)
        sprite.position.set(position.x, position.y, position.z)
        this.scene.add(sprite)
        const transformControls = new TransformControls(
          this.camera, this.renderer.domElement
        )
        transformControls.addEventListener(
          'mouseDown', function (e) {
            this.controls.enablePan = false
            this.controls.enableRotate = false
          }.bind(this)
        )
        transformControls.addEventListener(
          'mouseUp', function (e) {
            this.controls.enablePan = true
            this.controls.enableRotate = true
          }.bind(this)
        )
        // transformControls.showX = false
        // transformControls.showY = false
        transformControls.showZ = false
        transformControls.attach(sprite)
        this.scene.add(transformControls)
      }

      const createCanvasForCompany = (canvasWidth, canvasHeight, text, fontSize) => {
        // 貼り付けるcanvasを作成。
        const canvasForText = document.createElement('canvas')
        const ctx = canvasForText.getContext('2d')
        ctx.canvas.width = canvasWidth
        ctx.canvas.height = canvasHeight
        // 透過率50%の青背景を描く
        ctx.fillStyle = 'rgba(0, 0, 255, 0.5)'
        ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height)
        //
        ctx.fillStyle = 'black'
        ctx.font = `${fontSize}px serif`
        ctx.fillText(
          text,
          // x方向の余白/2をx方向開始時の始点とすることで、横方向の中央揃えをしている。
          (canvasWidth - ctx.measureText(text).width) / 2,
          // y方向のcanvasの中央に文字の高さの半分を加えることで、縦方向の中央揃えをしている。
          canvasHeight / 2 + ctx.measureText(text).actualBoundingBoxAscent / 2
        )
        return canvasForText
      }

      const createCanvasForTerm = (canvasWidth, canvasHeight, text, fontSize) => {
        // 貼り付けるcanvasを作成。
        const canvasForText = document.createElement('canvas')
        const ctx = canvasForText.getContext('2d')
        ctx.canvas.width = canvasWidth
        ctx.canvas.height = canvasHeight
        // 透過率50%の青背景を描く
        ctx.fillStyle = 'rgba(255, 0, 0, 0.5)'
        ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height)
        //
        ctx.fillStyle = 'black'
        ctx.font = `${fontSize}px serif`
        ctx.fillText(
          text,
          // x方向の余白/2をx方向開始時の始点とすることで、横方向の中央揃えをしている。
          (canvasWidth - ctx.measureText(text).width) / 2,
          // y方向のcanvasの中央に文字の高さの半分を加えることで、縦方向の中央揃えをしている。
          canvasHeight / 2 + ctx.measureText(text).actualBoundingBoxAscent / 2
        )
        return canvasForText
      }
      const canvasWidth = 400
      const canvasHeight = 40
      const scaleMaster = 70

      const companyObjList = company
      const termObjList = term
      const objects = []

      companyObjList.forEach((element) => {
        const texture = new THREE.CanvasTexture(
          createCanvasForCompany(canvasWidth, canvasHeight, element.company, 30)
        )
        createSprite(
          texture,
          {
            x: scaleMaster,
            // 縦方向の縮尺を調整
            y: scaleMaster * (canvasHeight / canvasWidth),
            z: scaleMaster
          },
          element.coordinate
        )
        objects.push(texture)
      })

      termObjList.forEach((element) => {
        const texture = new THREE.CanvasTexture(
          createCanvasForTerm(canvasWidth, canvasHeight, element.term, 30)
        )
        createSprite(
          texture,
          {
            x: scaleMaster,
            // 縦方向の縮尺を調整
            y: scaleMaster * (canvasHeight / canvasWidth),
            z: scaleMaster
          },
          element.coordinate
        )
        objects.push(texture)
      })
      // この平面に対してオブジェクトを平行に動かす
      var plane = new THREE.Plane()

      var raycaster = new THREE.Raycaster()
      var mouse = new THREE.Vector2()
      var offset = new THREE.Vector3()
      var intersection = new THREE.Vector3()

      // マウスオーバーしているオブジェクト
      var mouseoveredObj
      // ドラッグしているオブジェクト
      var draggedObj

      this.renderer.domElement.addEventListener('mousedown', onDocumentMouseDown, false)
      this.renderer.domElement.addEventListener('mousemove', onDocumentMouseMove, false)
      this.renderer.domElement.addEventListener('mouseup', onDocumentMouseUp, false)

      function onDocumentMouseDown (event) {
        event.preventDefault()

        raycaster.setFromCamera(mouse, this.camera)
        var intersects = raycaster.intersectObjects(objects)

        if (intersects.length > 0) {
        // マウスドラッグしている間はTrackballControlsを無効にする
          this.controls.enabled = false

          draggedObj = intersects[0].object

          // rayとplaneの交点を求めてintersectionに設定
          if (raycaster.ray.intersectPlane(plane, intersection)) {
          // ドラッグ中のオブジェクトとplaneの距離
            offset.copy(intersection).sub(draggedObj.position)
          }
        }
      }

      function onDocumentMouseMove (event) {
        event.preventDefault()

        mouse.x = (event.clientX / window.innerWidth) * 2 - 1
        mouse.y = -(event.clientY / window.innerHeight) * 2 + 1

        raycaster.setFromCamera(mouse, this.camera)

        if (draggedObj) {
        // オブジェクトをドラッグして移動させているとき

          // rayとplaneの交点をintersectionに設定
          if (raycaster.ray.intersectPlane(plane, intersection)) {
          // オブジェクトをplaneに対して平行に移動させる
            draggedObj.position.copy(intersection.sub(offset))
          }
        } else {
        // オブジェクトをドラッグしないでマウスを動かしている場合
          var intersects = raycaster.intersectObjects(objects)

          if (intersects.length > 0) {
            if (mouseoveredObj !== intersects[0].object) {
              // マウスオーバー中のオブジェクトを入れ替え
              mouseoveredObj = intersects[0].object

              // plane.normalにカメラの方向ベクトルを設定
              // 平面の角度をカメラの向きに対して垂直に維持する
              this.camera.getWorldDirection(plane.normal)
            }
          } else {
            mouseoveredObj = null
          }
        }
      }

      function onDocumentMouseUp (event) {
        event.preventDefault()

        this.controls.enabled = true

        if (mouseoveredObj) {
          draggedObj = null
        }
      }
    },
    animate () {
      requestAnimationFrame(this.animate)

      // this.cube.rotation.x += 0.01
      // this.cube.rotation.y += 0.01

      this.renderer.render(this.scene, this.camera)
    }
  }
}
</script>

<style scoped>
  .fullscreen {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
  }
  </style>
