import Highcharts from 'highcharts'
import More from 'highcharts/highcharts-more'
import draggablePoints from 'highcharts/modules/draggable-points'

More(Highcharts)
draggablePoints(Highcharts)
const companyTableData = {
  company: ''
}

const termTableData = {
  term: ''
}

const updateCompanyIndex = []
const updateTermIndex = []

const chartOptions = {
  chart: {
    width: Math.min(window.innerHeight, window.innerWidth) * 1.0,
    height: 100 + '%',
    zoomType: 'xy'
  },
  tooltip: {
    useHTML: true,
    formatter: function () {
      return this.point.label
    }
  },
  xAxis: {
    // min: -1,
    // max: 1,
    gridLineWidth: 1,
    minorTickInterval: 0.1,
    tickInterval: 0.2
  },
  yAxis: {
    // min: -1,
    // max: 1,
    minorTickInterval: 0.1,
    tickInterval: 0.2
  },
  legend: {
    layout: 'vertical',
    align: 'left',
    verticalAlign: 'top',
    floating: true,
    backgroundColor: Highcharts.defaultOptions.chart.backgroundColor,
    borderWidth: 1
  },
  title: {
    text: '潜在空間'
  },
  series: [
    {
      name: 'Company',
      data: [],
      dataLabal: [],
      type: 'scatter',
      animation: false,
      dragDrop: {
        draggableX: true,
        draggableY: true,
        liveRedraw: true
      },
      point: {
        events: {
          drop: function (e) {
            const point = this
            // point.index は数値指定で点を移動させると変わってしまうので独自のdataIndexを使用
            const index = point.dataIndex
            if (e.newPoint.x !== undefined) {
              chartOptions.series[0].data[index].x = e.newPoint.x
              chartOptions.series[0].data[index].y = e.newPoint.y
            }
            updateCompanyIndex.push(index)
          }
        }
      }
    },
    {
      name: 'Term',
      data: [],
      dataLabal: [],
      type: 'scatter',
      color: 'red',
      animation: false,
      dragDrop: {
        draggableX: true,
        draggableY: true,
        liveRedraw: true
      },
      point: {
        events: {
          drop: function (e) {
            const point = this
            // point.index は数値指定で点を移動させると変わってしまうので独自のdataIndexを使用
            const index = point.dataIndex
            if (e.newPoint.x !== undefined) {
              chartOptions.series[1].data[index].x = e.newPoint.x
              chartOptions.series[1].data[index].y = e.newPoint.y
            }
            updateTermIndex.push(index)
          }
        }
      }
    }
  ],
  plotOptions: {
    series: {
      states: {
        hover: {
          enabled: false
        }
      },

      dataLabels: {
        enabled: true,
        allowOverlap: true,
        format: '{point.company}{point.term}'
      }
    }
  }
}

export {
  companyTableData,
  termTableData,
  chartOptions,
  updateCompanyIndex,
  updateTermIndex
}
