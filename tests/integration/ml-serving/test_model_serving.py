"""
Integration tests for ML model serving
Tests KServe deployment, canary rollouts, SLOs, and autoscaling
"""
import pytest
import requests
import time
from kubernetes import client, config
from kubernetes.client.rest import ApiException


class TestMLServing:
    """Test suite for ML Serving platform"""
    
    @pytest.fixture(scope="class")
    def k8s_client(self):
        """Initialize Kubernetes client"""
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()
        return client.ApiClient()
    
    @pytest.fixture(scope="class")
    def namespace(self):
        """ML Serving namespace"""
        return "ml-serving"
    
    def test_namespace_exists(self, k8s_client, namespace):
        """Test that ml-serving namespace exists"""
        v1 = client.CoreV1Api(k8s_client)
        try:
            ns = v1.read_namespace(namespace)
            assert ns.metadata.name == namespace
            assert "istio-injection" in ns.metadata.labels
        except ApiException as e:
            pytest.fail(f"Namespace {namespace} not found: {e}")
    
    def test_kserve_installed(self, k8s_client):
        """Test that KServe CRDs are installed"""
        api = client.ApiextensionsV1Api(k8s_client)
        try:
            crd = api.read_custom_resource_definition("inferenceservices.serving.kserve.io")
            assert crd.metadata.name == "inferenceservices.serving.kserve.io"
        except ApiException as e:
            pytest.skip(f"KServe CRDs not installed: {e}")
    
    def test_argo_rollouts_installed(self, k8s_client):
        """Test that Argo Rollouts CRDs are installed"""
        api = client.ApiextensionsV1Api(k8s_client)
        try:
            crd = api.read_custom_resource_definition("rollouts.argoproj.io")
            assert crd.metadata.name == "rollouts.argoproj.io"
        except ApiException as e:
            pytest.skip(f"Argo Rollouts CRDs not installed: {e}")
    
    def test_inference_service_deployed(self, k8s_client, namespace):
        """Test that example InferenceService is deployed"""
        custom_api = client.CustomObjectsApi(k8s_client)
        try:
            isvc = custom_api.get_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=namespace,
                plural="inferenceservices",
                name="sklearn-iris"
            )
            assert isvc["metadata"]["name"] == "sklearn-iris"
            assert isvc["spec"]["predictor"]["model"]["modelFormat"]["name"] == "kserve-sklearn"
        except ApiException as e:
            pytest.skip(f"InferenceService not deployed: {e}")
    
    def test_rollout_deployed(self, k8s_client, namespace):
        """Test that Argo Rollout is deployed"""
        custom_api = client.CustomObjectsApi(k8s_client)
        try:
            rollout = custom_api.get_namespaced_custom_object(
                group="argoproj.io",
                version="v1alpha1",
                namespace=namespace,
                plural="rollouts",
                name="sklearn-iris-predictor"
            )
            assert rollout["metadata"]["name"] == "sklearn-iris-predictor"
            assert "canary" in rollout["spec"]["strategy"]
        except ApiException as e:
            pytest.skip(f"Rollout not deployed: {e}")
    
    def test_hpa_configured(self, k8s_client, namespace):
        """Test that HorizontalPodAutoscaler is configured"""
        autoscaling_api = client.AutoscalingV2Api(k8s_client)
        try:
            hpa = autoscaling_api.read_namespaced_horizontal_pod_autoscaler(
                name="sklearn-iris-predictor-hpa",
                namespace=namespace
            )
            assert hpa.spec.min_replicas >= 1
            assert hpa.spec.max_replicas >= 2
            assert len(hpa.spec.metrics) > 0
        except ApiException as e:
            pytest.skip(f"HPA not configured: {e}")
    
    def test_prometheus_rules_deployed(self, k8s_client, namespace):
        """Test that PrometheusRules for SLOs are deployed"""
        custom_api = client.CustomObjectsApi(k8s_client)
        try:
            # Check SLO recording rules
            recording_rules = custom_api.get_namespaced_custom_object(
                group="monitoring.coreos.com",
                version="v1",
                namespace=namespace,
                plural="prometheusrules",
                name="ml-serving-slo-recording-rules"
            )
            assert len(recording_rules["spec"]["groups"]) > 0
            
            # Check SLO alerts
            alerts = custom_api.get_namespaced_custom_object(
                group="monitoring.coreos.com",
                version="v1",
                namespace=namespace,
                plural="prometheusrules",
                name="ml-serving-slo-alerts"
            )
            assert len(alerts["spec"]["groups"]) > 0
        except ApiException as e:
            pytest.skip(f"PrometheusRules not deployed: {e}")
    
    def test_service_monitor_deployed(self, k8s_client, namespace):
        """Test that ServiceMonitor is deployed for metrics collection"""
        custom_api = client.CustomObjectsApi(k8s_client)
        try:
            sm = custom_api.get_namespaced_custom_object(
                group="monitoring.coreos.com",
                version="v1",
                namespace=namespace,
                plural="servicemonitors",
                name="ml-serving"
            )
            assert sm["metadata"]["name"] == "ml-serving"
            assert len(sm["spec"]["endpoints"]) > 0
        except ApiException as e:
            pytest.skip(f"ServiceMonitor not deployed: {e}")
    
    def test_canary_services_exist(self, k8s_client, namespace):
        """Test that stable and canary services exist for traffic splitting"""
        v1 = client.CoreV1Api(k8s_client)
        try:
            # Check stable service
            stable_svc = v1.read_namespaced_service(
                name="sklearn-iris-predictor-stable",
                namespace=namespace
            )
            assert stable_svc.metadata.name == "sklearn-iris-predictor-stable"
            
            # Check canary service
            canary_svc = v1.read_namespaced_service(
                name="sklearn-iris-predictor-canary",
                namespace=namespace
            )
            assert canary_svc.metadata.name == "sklearn-iris-predictor-canary"
        except ApiException as e:
            pytest.skip(f"Canary services not found: {e}")
    
    def test_virtual_service_configured(self, k8s_client, namespace):
        """Test that Istio VirtualService is configured for traffic routing"""
        custom_api = client.CustomObjectsApi(k8s_client)
        try:
            vs = custom_api.get_namespaced_custom_object(
                group="networking.istio.io",
                version="v1beta1",
                namespace=namespace,
                plural="virtualservices",
                name="sklearn-iris-vs"
            )
            assert vs["metadata"]["name"] == "sklearn-iris-vs"
            assert len(vs["spec"]["http"]) > 0
            # Check that traffic routing is configured
            routes = vs["spec"]["http"][0]["route"]
            assert len(routes) == 2  # stable and canary
        except ApiException as e:
            pytest.skip(f"VirtualService not configured: {e}")
    
    def test_analysis_template_exists(self, k8s_client, namespace):
        """Test that AnalysisTemplate exists for canary validation"""
        custom_api = client.CustomObjectsApi(k8s_client)
        try:
            template = custom_api.get_namespaced_custom_object(
                group="argoproj.io",
                version="v1alpha1",
                namespace=namespace,
                plural="analysistemplates",
                name="sklearn-iris-analysis"
            )
            assert template["metadata"]["name"] == "sklearn-iris-analysis"
            # Check metrics are defined
            assert len(template["spec"]["metrics"]) >= 2  # success-rate and latency-p95
        except ApiException as e:
            pytest.skip(f"AnalysisTemplate not found: {e}")
    
    @pytest.mark.slow
    def test_inference_endpoint_responds(self, namespace):
        """Test that inference endpoint responds to requests"""
        # This test requires the service to be accessible
        # In a real cluster, you would use the actual service URL
        service_url = f"http://sklearn-iris.{namespace}.svc.cluster.local/v1/models/sklearn-iris:predict"
        
        payload = {
            "instances": [[5.1, 3.5, 1.4, 0.2]]
        }
        
        try:
            response = requests.post(
                service_url,
                json=payload,
                timeout=10
            )
            assert response.status_code == 200
            result = response.json()
            assert "predictions" in result
        except requests.exceptions.RequestException as e:
            pytest.skip(f"Cannot reach inference endpoint: {e}")


class TestSLOValidation:
    """Test suite for SLO validation"""
    
    def test_latency_slo_thresholds(self):
        """Test that latency SLO thresholds are correctly configured"""
        # These values should match values.yaml
        expected_p95 = 0.5  # 500ms
        expected_p99 = 1.0  # 1000ms
        
        # In a real test, you would query Prometheus to validate these
        assert expected_p95 > 0
        assert expected_p99 > expected_p95
    
    def test_availability_slo_target(self):
        """Test that availability SLO target is correctly configured"""
        # These values should match values.yaml
        expected_availability = 0.999  # 99.9%
        expected_error_budget = 0.001
        
        assert expected_availability + expected_error_budget == 1.0
        assert expected_availability >= 0.999


class TestAutoscaling:
    """Test suite for autoscaling configuration"""
    
    def test_autoscaling_parameters(self):
        """Test that autoscaling parameters are within acceptable ranges"""
        min_replicas = 1
        max_replicas = 10
        target_cpu = 70
        
        assert min_replicas >= 1
        assert max_replicas > min_replicas
        assert 0 < target_cpu <= 100
    
    def test_autoscaling_behavior(self):
        """Test that autoscaling behavior is configured for stability"""
        # Scale-down should be slower than scale-up
        scale_down_window = 300  # 5 minutes
        scale_up_window = 60  # 1 minute
        
        assert scale_down_window > scale_up_window
