import os

from aws_cdk import (
    App,
    CfnOutput,
    Duration,
    Environment,
    Stack,
    aws_lambda,
    aws_iam
)
from constructs import Construct




class GradioLambdaFnStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # https://github.com/aws-samples/aws-cdk-examples/blob/main/python/lambda-from-container/app.py
        ecr_image = aws_lambda.EcrImageCode.from_asset_image(
            directory=os.path.join(os.getcwd()), file="Dockerfile.lambdafn"
        )

        # from dotenv import load_dotenv
        # load_dotenv()  # dockerfile exposed to Secret Keys 😨
        # DockerImage.from_build(path= os.getcwd()  ,cache_disabled=True,file='Dockerfile.lambdafn',build_args={'AWS_ACCESS_KEY_ID':"",'AWS_SECRET_x1ACCESS_KEY':'','CDK_DEFAULT_ACCOUNT':'',"CDK_DEFAULT_REGION":''})

        # IN CLI,do::  `aws ecr describe-repositories`

        # image_name = ""
        # ecr_repository = aws_ecr.Repository.from_repository_attributes(self,
        #         id              = "ECR",
        #         repository_arn  ='arn:aws:ecr:{0}:{1}:repository/{2}'.format(Aws.REGION, Aws.ACCOUNT_ID, image_name),
        #         repository_name = image_name
        #     )
        # ecr_image = typing.cast("aws_lambda.Code", aws_lambda.EcrImageCode( repository = ecr_repository ))
        lambda_role = aws_iam.Role(
            self, "LambdaExecutionRole",
            assumed_by=aws_iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                aws_iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3ReadOnlyAccess")
            ]
        )
        lambda_fn = aws_lambda.Function(
            self,
            id="GradioLambdaFn",
            description="Sample Lambda Container Function",
            code=ecr_image,
            handler=aws_lambda.Handler.FROM_IMAGE,
            runtime=aws_lambda.Runtime.FROM_IMAGE,
            function_name="emloLambdaFunction",
            memory_size=3008,
            timeout=Duration.seconds(60),
            role=lambda_role,
            # environment = {
            #     AWS_ACCESS_KEY: os.getenv("AWS_ACCESS_KEY"),  # AWS.ACCOUNT_ID
            #     AWS_SECRET_KEY: os.getenv("AWS_SECRET_KEY"),  # AWS.REGION
            #     AWS_REGION: os.getenv("AWS_REGION"),  # AWS.ACCOUNT_ID
            # }
        )

        # Add function URL
        fn_url = lambda_fn.add_function_url(
            auth_type=aws_lambda.FunctionUrlAuthType.NONE
        )

        # Output the function URL
        CfnOutput(
            self, id="functionURL", value=fn_url.url, description="Lambda Function URL"
        )


# Initialize the CDK app
app = App(analytics_reporting=True, auto_synth=False)


# Define environment
env = Environment(
    account=os.getenv("CDK_DEFAULT_ACCOUNT"),  # AWS.ACCOUNT_ID
    region=os.getenv("CDK_DEFAULT_REGION", "ap-south-1"),  # AWS.REGION

)

# Create the stack
GradioLambdaFnStack(app, "GradioLambdaFnStack", env=env)


app.synth() # make auto_synth=True