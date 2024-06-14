from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from requests_aws4auth import AWS4Auth
import boto3


def build_local_docker_connection() -> OpenSearch:
    """
    Build a connection to a local docker container running OpenSearch.
    """
    return OpenSearch(
        hosts=[{'host': 'localhost', 'port': '9200'}],
        use_ssl=True,
        verify_certs=False,
        http_auth=('admin', 'admin'),
        ssl_assert_hostname=False,
        ssl_show_warn=False
    )


def build_aws_search_service(stack_name: str, application_prefix: str) -> OpenSearch:
    """
    Build a connection to an AWS OpenSearch service. This part expects to use the setup according to my CDK stack.
    """
    session = boto3.Session(profile_name='default')
    cfn = session.client('cloudformation')
    response = cfn.describe_stacks(
        StackName=stack_name
    )
    outputs = {
        output['ExportName']: output['OutputValue']
        for output in response['Stacks'][0]['Outputs']
        if 'ExportName' in output
    }
    host = 'https://' + outputs[f'{application_prefix}-DomainEndpoint']
    port = 443
    region = 'eu-west-1'  # e.g. us-west-1

    sts = session.client('sts')
    response = sts.assume_role(
        RoleArn=outputs[f'{application_prefix}-AdminUserRoleArn'],
        RoleSessionName="assumed-opensearch-user-admin-role",
    )
    auth = AWS4Auth(
        response['Credentials']['AccessKeyId'], response['Credentials']['SecretAccessKey'],
        region, 'es',
        session_token=response['Credentials']['SessionToken']
    )
    return OpenSearch(
        hosts=[f'{host}:{port}'],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )
