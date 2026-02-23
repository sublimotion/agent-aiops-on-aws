# Monitoring Module - Prometheus and Grafana for vLLM metrics collection

resource "kubernetes_namespace" "monitoring" {
  metadata {
    name = var.namespace
    labels = {
      name = var.namespace
    }
  }
}

# Prometheus with 1s scrape interval for vLLM metrics
resource "helm_release" "prometheus" {
  name       = "prometheus"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "prometheus"
  namespace  = kubernetes_namespace.monitoring.metadata[0].name
  version    = var.prometheus_version

  values = [
    yamlencode({
      server = {
        global = {
          scrape_interval     = var.scrape_interval
          evaluation_interval = var.evaluation_interval
        }
        persistentVolume = {
          enabled      = var.enable_persistence
          size         = var.prometheus_storage_size
          storageClass = var.storage_class
        }
        service = {
          type = "ClusterIP"
        }
      }

      alertmanager = {
        enabled = var.enable_alertmanager
      }

      # Additional scrape configs for vLLM
      serverFiles = {
        "prometheus.yml" = {
          scrape_configs = [
            {
              job_name        = "kubernetes-pods"
              scrape_interval = var.scrape_interval
              kubernetes_sd_configs = [
                {
                  role = "pod"
                }
              ]
              relabel_configs = [
                {
                  source_labels = ["__meta_kubernetes_pod_annotation_prometheus_io_scrape"]
                  action        = "keep"
                  regex         = "true"
                },
                {
                  source_labels = ["__meta_kubernetes_pod_annotation_prometheus_io_path"]
                  action        = "replace"
                  target_label  = "__metrics_path__"
                  regex         = "(.+)"
                },
                {
                  source_labels = ["__address__", "__meta_kubernetes_pod_annotation_prometheus_io_port"]
                  action        = "replace"
                  regex         = "([^:]+)(?::\\d+)?;(\\d+)"
                  replacement   = "$1:$2"
                  target_label  = "__address__"
                },
                {
                  action = "labelmap"
                  regex  = "__meta_kubernetes_pod_label_(.+)"
                },
                {
                  source_labels = ["__meta_kubernetes_namespace"]
                  action        = "replace"
                  target_label  = "kubernetes_namespace"
                },
                {
                  source_labels = ["__meta_kubernetes_pod_name"]
                  action        = "replace"
                  target_label  = "kubernetes_pod_name"
                }
              ]
            },
            {
              job_name        = "vllm"
              scrape_interval = var.scrape_interval
              static_configs = [
                {
                  targets = var.vllm_targets
                }
              ]
            }
          ]
        }
      }
    })
  ]
}

# Grafana
resource "helm_release" "grafana" {
  name       = "grafana"
  repository = "https://grafana.github.io/helm-charts"
  chart      = "grafana"
  namespace  = kubernetes_namespace.monitoring.metadata[0].name
  version    = var.grafana_version

  values = [
    yamlencode({
      adminPassword = var.grafana_admin_password

      persistence = {
        enabled          = var.enable_persistence
        size             = var.grafana_storage_size
        storageClassName = var.storage_class
      }

      service = {
        type     = var.grafana_service_type
        nodePort = var.grafana_service_type == "NodePort" ? var.grafana_node_port : null
      }

      datasources = {
        "datasources.yaml" = {
          apiVersion = 1
          datasources = [
            {
              name      = "Prometheus"
              type      = "prometheus"
              url       = "http://prometheus-server:80"
              access    = "proxy"
              isDefault = true
            }
          ]
        }
      }

      # vLLM dashboard
      dashboardProviders = {
        "dashboardproviders.yaml" = {
          apiVersion = 1
          providers = [
            {
              name            = "default"
              orgId           = 1
              folder          = ""
              type            = "file"
              disableDeletion = false
              editable        = true
              options = {
                path = "/var/lib/grafana/dashboards/default"
              }
            }
          ]
        }
      }

      dashboardsConfigMaps = {
        default = "grafana-dashboards"
      }
    })
  ]

  depends_on = [helm_release.prometheus]
}

# vLLM Dashboard ConfigMap
resource "kubernetes_config_map" "grafana_dashboards" {
  metadata {
    name      = "grafana-dashboards"
    namespace = kubernetes_namespace.monitoring.metadata[0].name
  }

  data = {
    "vllm-dashboard.json" = jsonencode({
      title = "vLLM KV Cache Benchmark"
      uid   = "vllm-kv-bench"
      panels = [
        {
          title      = "GPU Memory Usage"
          type       = "timeseries"
          gridPos    = { h = 8, w = 12, x = 0, y = 0 }
          datasource = { type = "prometheus", uid = "prometheus" }
          targets = [
            {
              expr         = "vllm:gpu_cache_usage_perc"
              legendFormat = "KV Cache %"
            }
          ]
        },
        {
          title      = "Prefix Cache Hit Rate"
          type       = "timeseries"
          gridPos    = { h = 8, w = 12, x = 12, y = 0 }
          datasource = { type = "prometheus", uid = "prometheus" }
          targets = [
            {
              expr         = "rate(vllm:prefix_cache_hit_total[1m]) / (rate(vllm:prefix_cache_hit_total[1m]) + rate(vllm:prefix_cache_miss_total[1m]))"
              legendFormat = "Hit Rate"
            }
          ]
        },
        {
          title      = "Preemptions"
          type       = "timeseries"
          gridPos    = { h = 8, w = 12, x = 0, y = 8 }
          datasource = { type = "prometheus", uid = "prometheus" }
          targets = [
            {
              expr         = "rate(vllm:num_preemptions_total[1m])"
              legendFormat = "Preemptions/sec"
            }
          ]
        },
        {
          title      = "Request Throughput"
          type       = "timeseries"
          gridPos    = { h = 8, w = 12, x = 12, y = 8 }
          datasource = { type = "prometheus", uid = "prometheus" }
          targets = [
            {
              expr         = "rate(vllm:request_success_total[1m])"
              legendFormat = "Requests/sec"
            }
          ]
        }
      ]
      schemaVersion = 38
      time          = { from = "now-15m", to = "now" }
      refresh       = "5s"
    })
  }

  depends_on = [kubernetes_namespace.monitoring]
}
